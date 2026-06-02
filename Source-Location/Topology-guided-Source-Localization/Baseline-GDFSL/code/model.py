import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import math
from GCN_EN import self_loop_attention_GCN


def extract(tensor, t, shape):
    tensor_t = tensor[t]  
    
    return tensor_t


import torch
import math

def get_time_embedding(t, dimension, device):    
    position = torch.tensor([t], dtype=torch.float32, device=device).unsqueeze(1)   
    div_term = torch.exp(torch.arange(0, dimension, 2, device=device) * -(math.log(10000.0) / dimension))    
    pos_embedding = torch.zeros((1, dimension), device=device)    
    pos_embedding[:, 0::2] = torch.sin(position * div_term)  
    pos_embedding[:, 1::2] = torch.cos(position * div_term)  
    return pos_embedding


class DiffusionModule_predict_X0(nn.Module):    
    def __init__(self, feature_hidden_dim=64, x_in_dim=1, feature_dim=14, pro_dyn_feature_dim=7, head=2, self_head=4):
        super(DiffusionModule_predict_X0, self).__init__()
        self.feature_hidden_dim = feature_hidden_dim
        
        self.MLP = nn.Linear(x_in_dim, feature_hidden_dim)
        self.norm1 = nn.LayerNorm(feature_hidden_dim)
        
        self.high_focus1 = nn.Parameter(torch.Tensor(pro_dyn_feature_dim, feature_hidden_dim // 2))
        self.high_bias1 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2))
        self.high_focus2 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2, feature_hidden_dim))
        self.high_bias2 = nn.Parameter(torch.Tensor(feature_hidden_dim))
        self.high_norm1 = nn.LayerNorm(feature_hidden_dim // 2)
        self.high_norm2 = nn.LayerNorm(feature_hidden_dim)

        
        self.low_focus1 = nn.Parameter(torch.Tensor(pro_dyn_feature_dim, feature_hidden_dim // 2))
        self.low_bias1 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2))
        self.low_focus2 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2, feature_hidden_dim))
        self.low_bias2 = nn.Parameter(torch.Tensor(feature_hidden_dim))
        self.low_norm1 = nn.LayerNorm(feature_hidden_dim // 2)
        self.low_norm2 = nn.LayerNorm(feature_hidden_dim)

        
        self.crossAtt = nn.MultiheadAttention(feature_hidden_dim, num_heads=head)
        self.cross_norm = nn.LayerNorm(feature_hidden_dim)
        
        self.loop_att_gcn = self_loop_attention_GCN(feature_dim, self_head)

        self.de_Conv = nn.Parameter(torch.Tensor(feature_hidden_dim, feature_dim))
        self.de_bias = nn.Parameter(torch.Tensor(feature_dim))
        self.de_norm1 = nn.LayerNorm(feature_dim)

        self.de_MLP = nn.Linear(2 * feature_dim, 1)

        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.high_focus1)
        nn.init.xavier_uniform_(self.low_focus1)
        nn.init.xavier_uniform_(self.high_focus2)
        nn.init.xavier_uniform_(self.low_focus2)
        nn.init.constant_(self.high_bias1, 0)
        nn.init.constant_(self.low_bias1, 0)
        nn.init.constant_(self.high_bias2, 0)
        nn.init.constant_(self.low_bias2, 0)

        nn.init.xavier_uniform_(self.de_Conv)
        nn.init.constant_(self.de_bias, 0)


    def forward(self, x_t, timestamp, pro_dyn_feature, graph_topo, features):  
        assert x_t.shape[0]==pro_dyn_feature.shape[0]==graph_topo.shape[0], 'error'
        
        position_em_hidden = get_time_embedding(timestamp, self.feature_hidden_dim, x_t.device)                  
        x_t_hidden = self.MLP(x_t)
        x_t_hidden = self.norm1(x_t_hidden)  

        
        graph_topo = graph_topo.squeeze(0)
        laplacian = torch.diag(torch.sum(graph_topo, dim=1)) - graph_topo
        high_freq = torch.matmul(laplacian, pro_dyn_feature)
        high_freq = torch.matmul(high_freq, self.high_focus1) + self.high_bias1
        high_freq = self.high_norm1(high_freq)
        high_freq = F.relu(high_freq)

        high_freq = torch.matmul(high_freq, self.high_focus2) + self.high_bias2
        high_freq = self.high_norm2(high_freq)
        high_freq = F.relu(high_freq)   

        
        D = torch.diag(torch.pow(graph_topo.sum(dim=1), -0.5))
        D[torch.isinf(D)] = 0
        A_hat = graph_topo + torch.eye(graph_topo.size(0)).cuda()
        DAD = torch.mm(torch.mm(D, A_hat), D)
        low_freq = torch.matmul(DAD, pro_dyn_feature)
        low_freq = torch.matmul(low_freq, self.low_focus1) + self.low_bias1
        low_freq = self.low_norm1(low_freq)
        low_freq = F.relu(low_freq)

        low_freq = torch.matmul(low_freq, self.low_focus2) + self.low_bias2
        low_freq = self.low_norm2(low_freq)
        low_freq = F.relu(low_freq) 

        
        high_freq = high_freq.unsqueeze(0)
        low_freq = low_freq.unsqueeze(0)
        cross_att_output, _ = self.crossAtt(high_freq, low_freq, low_freq)
        cross_att_output = cross_att_output.squeeze(0)
        cross_att_output = self.cross_norm(cross_att_output) 
        encoder_input = x_t_hidden + position_em_hidden + cross_att_output

        encoder_DAD = torch.matmul(DAD, encoder_input)
        encoder_output = torch.matmul(encoder_DAD, self.de_Conv) + self.de_bias  
        encoder_output = self.de_norm1(encoder_output)
        
        encoder_output = encoder_output + pro_dyn_feature[:, -2].unsqueeze(1)

        enhanced_fea = self.loop_att_gcn(graph_topo, features)
        decoder_input = torch.cat((encoder_output, enhanced_fea), dim=-1)

        decoder_output = self.de_MLP(decoder_input) + pro_dyn_feature[:, -2].unsqueeze(1)

        return decoder_output

class GaussianDiffusionForwardTrainer(nn.Module):
    def __init__(self, beta_1, beta_T, T, model):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('one_minus_sqrt_alphas_bar', 1. - torch.sqrt(alphas_bar))

    def forward_step(self, x_0, t, final_miu):
        assert x_0.shape == final_miu.shape, 'error'
        
        noise = torch.randn_like(x_0)
        x_t = self.sqrt_alphas_bar[t] * x_0 + \
              self.one_minus_sqrt_alphas_bar[t] * final_miu + \
              self.sqrt_one_minus_alphas_bar[t] * noise
        
        return x_t

class GaussianDiffusionForwardTrainer_future(nn.Module):
    def __init__(self, t_start, beta_1, beta_T, T, feature_hidden_dim):
        super().__init__()
        self.model = DiffusionModule_predict_X0(feature_hidden_dim)
        self.T = T
        self.t_start = t_start

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double()[t_start:])
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('one_minus_sqrt_alphas_bar', 1. - torch.sqrt(alphas_bar))

    def forward_step_interval(self, x_start, t_interval, final_miu):
        assert x_start.shape == final_miu.shape, 'error'
        
        noise = torch.randn_like(x_start)
        x_t = self.sqrt_alphas_bar[t_interval] * x_start + \
              self.one_minus_sqrt_alphas_bar[t_interval] * final_miu + \
              self.sqrt_one_minus_alphas_bar[t_interval] * noise
        
        return x_t, noise

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model.model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', (1.-alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar))
        self.register_buffer('coeff2', self.betas * torch.sqrt(alphas_bar_prev) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
        self.register_buffer('additional_term', (
                (1 - self.sqrt_alphas_bar_prev) * (1 - torch.sqrt(alphas)) * (1 - self.sqrt_alphas_bar) / (1 - alphas_bar)
        )) 

    def predict_xt_prev_mean_from_X0(self, x_t, t, X0):
        assert x_t.shape == X0.shape
        return (
                self.coeff1[t] * x_t + self.coeff2[t] * X0
        )

    def p_mean_variance(self, x_t, t, pro_dyn_feature, graph_topo, features, final_miu):
        X0_hat = self.model(x_t, t, pro_dyn_feature, graph_topo, features)
        xt_prev_mean = self.predict_xt_prev_mean_from_X0(x_t, t, X0_hat)
        xt_prev_mean += self.additional_term[t] * final_miu

        var = self.posterior_var[t]
        return xt_prev_mean, var

    def forward(self, x_T, final_miu, pro_dyn_feature, graph_topo, features):  
        x_t = x_T
        for time_step in reversed(range(self.T)):
            mean, var = self.p_mean_variance(x_t, time_step, pro_dyn_feature, graph_topo, features, final_miu)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
        x_0 = x_t
        return torch.clip(x_0, 0, 10)


class DiffusionModule_predict_X0_un(nn.Module):   
    def __init__(self, feature_hidden_dim=64, x_in_dim=1, feature_dim=14, pro_dyn_feature_dim=7, head=2, self_head=4):
        super(DiffusionModule_predict_X0_un, self).__init__()
        self.feature_hidden_dim = feature_hidden_dim

        
        self.MLP = nn.Linear(x_in_dim, feature_hidden_dim)
        self.norm1 = nn.LayerNorm(feature_hidden_dim)       
        self.high_focus1 = nn.Parameter(torch.Tensor(pro_dyn_feature_dim, feature_hidden_dim // 2))
        self.high_bias1 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2))
        self.high_focus2 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2, feature_hidden_dim))
        self.high_bias2 = nn.Parameter(torch.Tensor(feature_hidden_dim))
        self.high_norm1 = nn.LayerNorm(feature_hidden_dim // 2)
        self.high_norm2 = nn.LayerNorm(feature_hidden_dim)

        
        self.low_focus1 = nn.Parameter(torch.Tensor(pro_dyn_feature_dim, feature_hidden_dim // 2))
        self.low_bias1 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2))
        self.low_focus2 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2, feature_hidden_dim))
        self.low_bias2 = nn.Parameter(torch.Tensor(feature_hidden_dim))
        self.low_norm1 = nn.LayerNorm(feature_hidden_dim // 2)
        self.low_norm2 = nn.LayerNorm(feature_hidden_dim)

        
        self.crossAtt = nn.MultiheadAttention(feature_hidden_dim, num_heads=head)
        self.cross_norm = nn.LayerNorm(feature_hidden_dim)

        
        
        self.loop_att_gcn = self_loop_attention_GCN(feature_dim, self_head)

        self.de_Conv = nn.Parameter(torch.Tensor(feature_hidden_dim, feature_dim))
        self.de_bias = nn.Parameter(torch.Tensor(feature_dim))
        self.de_norm1 = nn.LayerNorm(feature_dim)

        self.de_MLP1 = nn.Linear(2 * feature_dim, 1)
      
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.high_focus1)
        nn.init.xavier_uniform_(self.low_focus1)
        nn.init.xavier_uniform_(self.high_focus2)
        nn.init.xavier_uniform_(self.low_focus2)
        nn.init.constant_(self.high_bias1, 0)
        nn.init.constant_(self.low_bias1, 0)
        nn.init.constant_(self.high_bias2, 0)
        nn.init.constant_(self.low_bias2, 0)
        nn.init.xavier_uniform_(self.de_Conv)
        nn.init.constant_(self.de_bias, 0)


    def forward(self, x_t, timestamp, pro_dyn_feature, graph_topo, features):  
        assert x_t.shape[0]==pro_dyn_feature.shape[0]==graph_topo.shape[0], 'error'
        
        position_em_hidden = get_time_embedding(timestamp, self.feature_hidden_dim, x_t.device)  
                
        x_t_hidden = self.MLP(x_t)
        x_t_hidden = self.norm1(x_t_hidden)  

        
        graph_topo = graph_topo.squeeze(0)
        laplacian = torch.diag(torch.sum(graph_topo, dim=1)) - graph_topo
        high_freq = torch.matmul(laplacian, pro_dyn_feature)
        high_freq = torch.matmul(high_freq, self.high_focus1) + self.high_bias1
        high_freq = self.high_norm1(high_freq)
        high_freq = F.relu(high_freq)

        high_freq = torch.matmul(high_freq, self.high_focus2) + self.high_bias2
        high_freq = self.high_norm2(high_freq)
        high_freq = F.relu(high_freq)   

        
        D = torch.diag(torch.pow(graph_topo.sum(dim=1), -0.5))
        D[torch.isinf(D)] = 0
        A_hat = graph_topo + torch.eye(graph_topo.size(0)).cuda()
        DAD = torch.mm(torch.mm(D, A_hat), D)
        low_freq = torch.matmul(DAD, pro_dyn_feature)
        low_freq = torch.matmul(low_freq, self.low_focus1) + self.low_bias1
        low_freq = self.low_norm1(low_freq)
        low_freq = F.relu(low_freq)

        low_freq = torch.matmul(low_freq, self.low_focus2) + self.low_bias2
        low_freq = self.low_norm2(low_freq)
        low_freq = F.relu(low_freq) 
       
        high_freq = high_freq.unsqueeze(0)
        low_freq = low_freq.unsqueeze(0)
        cross_att_output, _ = self.crossAtt(high_freq, low_freq, low_freq)
        cross_att_output = cross_att_output.squeeze(0)
        cross_att_output = self.cross_norm(cross_att_output) 
        encoder_input = x_t_hidden + position_em_hidden + cross_att_output

        encoder_DAD = torch.matmul(DAD, encoder_input)
        encoder_output = torch.matmul(encoder_DAD, self.de_Conv) + self.de_bias  
        encoder_output = self.de_norm1(encoder_output)
        

        enhanced_fea = self.loop_att_gcn(graph_topo, features)
        decoder_input = torch.cat((encoder_output, enhanced_fea), dim=-1)

        decoder_output = self.de_MLP1(decoder_input)


        return decoder_output

class GaussianDiffusionForwardTrainer_future_un(nn.Module):
    def __init__(self, t_start, beta_1, beta_T, T, feature_hidden_dim):
        super().__init__()
        self.model = DiffusionModule_predict_X0_un(feature_hidden_dim)
        self.T = T
        self.t_start = t_start

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double()[t_start:])
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('one_minus_sqrt_alphas_bar', 1. - torch.sqrt(alphas_bar))

    def forward_step_interval(self, x_start, t_interval, final_miu):
        assert x_start.shape == final_miu.shape, 'error'
        
        noise = torch.randn_like(x_start)
        x_t = self.sqrt_alphas_bar[t_interval] * x_start + \
              self.one_minus_sqrt_alphas_bar[t_interval] * final_miu + \
              self.sqrt_one_minus_alphas_bar[t_interval] * noise
        
        return x_t, noise

    def forward_model_SI(self, x_start, t_interval, final_miu, adj, infection_rate=0.1):
        x = x_start.clone().float()  

        for _ in range(t_interval):
            infected = x.view(-1)  
            susceptible = 1.0 - infected              
            num_infected_neighbors = torch.matmul(adj, infected)                      
            infection_prob = 1.0 - torch.pow((1.0 - infection_rate), num_infected_neighbors)
            
            random_values = torch.rand(infection_prob.size(), device=x.device)
            new_infections = (random_values < infection_prob) * susceptible  
            
            x = torch.max(x, new_infections.view(-1, 1))
        
        x_t = x.int()

        noise = (x_t - self.sqrt_alphas_bar[t_interval] * x_start - self.one_minus_sqrt_alphas_bar[t_interval] * final_miu) \
                / self.sqrt_one_minus_alphas_bar[t_interval]

        return x_t, noise

    def forward_model_IC(self, x_start, t_interval, final_miu, adj, infection_rate=0.1):
        
        x = x_start.clone().float()  
        active_nodes = x.view(-1).clone()  

        for _ in range(t_interval):
            newly_active = torch.zeros_like(active_nodes)
            infected = x.view(-1)                        
            edges = adj * active_nodes.unsqueeze(1)              
            random_values = torch.rand(edges.size(), device=x.device)
            successful_infections = (random_values < infection_rate) * edges
            
            infection_attempts = torch.sum(successful_infections, dim=0)           
            new_infections = (infection_attempts > 0).float() * (1 - infected)            
            x = x + new_infections.view(-1, 1)
            
            active_nodes = new_infections
            
            if active_nodes.sum() == 0:
                break

        x_t = x.int()
       
        noise = (x_t - self.sqrt_alphas_bar[t_interval] * x_start - self.one_minus_sqrt_alphas_bar[
            t_interval] * final_miu) \
                / self.sqrt_one_minus_alphas_bar[t_interval]

        return x_t, noise



class GaussianDiffusionSampler_un(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, fairness=True):
        super().__init__()
        self.model = model.model
        self.T = T
        self.fair = fairness
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
        self.register_buffer('additional_term', (
                self.betas * torch.sqrt(alphas_bar_prev) * (1 - sqrt_alphas_bar) / (
                    sqrt_alphas_bar * (1 - alphas_bar)) - 
                (1 - sqrt_alphas_bar_prev) * (1 - torch.sqrt(alphas)) * (1 - sqrt_alphas_bar) / (1 - alphas_bar)
        ))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self.coeff1[t] * x_t - self.coeff2[t] * eps
        )

    def p_mean_variance(self, x_t, t, pro_dyn_feature, graph_topo, features, final_miu, observation):
        eps = self.model(x_t, t, pro_dyn_feature, graph_topo, features)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
        xt_prev_mean -= self.additional_term[t] * final_miu

        var = self.posterior_var[t]
        return 0.5 * xt_prev_mean + 0.5 * observation, var

    def forward(self, x_T, final_miu, pro_dyn_feature, graph_topo, features, observation):  
        x_t = x_T
        if self.fair:
            final_miu -= observation
        for time_step in reversed(range(self.T)):
            mean, var = self.p_mean_variance(x_t, time_step, pro_dyn_feature, graph_topo, features, final_miu, observation)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
        x_0 = x_t
        return torch.clip(x_0, 0, 10)
