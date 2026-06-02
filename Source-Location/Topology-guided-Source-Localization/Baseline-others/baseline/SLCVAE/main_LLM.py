import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

_BASELINE_OTHERS_DIR = Path(__file__).resolve().parents[2]
_TGS_ROOT = _BASELINE_OTHERS_DIR.parent
for _path in (_TGS_ROOT, _BASELINE_OTHERS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from baseline.SLCVAE.model_LLM import CVAE, GNN
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from utils import Metric
import pickle
import random

try:
    from project_paths import make_output_dir
    LOG_DIR = make_output_dir("logs")
except Exception:
    LOG_DIR = Path(__file__).resolve().parents[3] / "outputs" / "logs"


def _open_log(mode="a"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return open(LOG_DIR / "slcvae_llm.log", mode)


def _node_feature_dim(node_features):
    if node_features is None:
        return 1
    if torch.is_tensor(node_features):
        sample = node_features[0]
        if sample.dim() == 1:
            return 1
        return int(sample.shape[-1])

    feature_dims = []
    if node_features.get("content_features") is not None:
        sample = node_features["content_features"][0]
        feature_dims.append(1 if sample.dim() == 1 else int(sample.shape[-1]))
    if node_features.get("profile_embeddings") is not None:
        profile_feature = node_features["profile_embeddings"]
        feature_dims.append(1 if profile_feature.dim() == 1 else int(profile_feature.shape[-1]))
    if node_features.get("news_embeddings") is not None:
        sample = node_features["news_embeddings"][0]
        feature_dims.append(1 if sample.dim() == 1 else int(sample.shape[-1]))

    non_scalar_dims = [dim for dim in feature_dims if dim != 1]
    if not non_scalar_dims:
        return 1
    if len(set(non_scalar_dims)) != 1:
        raise ValueError(
            f"node feature dimensions must match for addition, got {feature_dims}")
    return non_scalar_dims[0]


def _add_node_feature_part(feature, part, name):
    if part.dim() == 1:
        part = part.unsqueeze(-1)
    if feature is None:
        return part
    if feature.shape[0] != part.shape[0]:
        raise ValueError(
            f"{name} has {part.shape[0]} nodes, expected {feature.shape[0]}")
    if feature.shape[-1] == part.shape[-1] or part.shape[-1] == 1:
        return feature + part
    if feature.shape[-1] == 1:
        return part + feature
    raise ValueError(
        f"{name} feature dim {part.shape[-1]} does not match existing dim {feature.shape[-1]}")


def _node_feature_for_sample(node_features, local_idx, device, seed_mask=None):
    if node_features is None:
        return None
    if torch.is_tensor(node_features):
        feature = node_features[local_idx]
        if feature.dim() == 1:
            feature = feature.unsqueeze(-1)
        return feature.to(device).float()

    feature = None
    content_feature = node_features.get("content_features")
    if content_feature is not None:
        content_feature = content_feature[local_idx]
        if content_feature.dim() == 1:
            content_feature = content_feature.unsqueeze(-1)
        content_feature = content_feature.to(device).float()
        feature = _add_node_feature_part(feature, content_feature, "content_features")

    news_feature = node_features.get("news_embeddings")
    if news_feature is not None:
        source_mask = node_features.get("source_masks")
        if source_mask is not None:
            source_mask = source_mask[local_idx]
        else:
            source_mask = seed_mask
        if source_mask is None:
            profile_feature = node_features.get("profile_embeddings")
            if feature is not None:
                num_nodes = feature.shape[0]
            elif profile_feature is not None:
                num_nodes = profile_feature.shape[0]
            else:
                raise ValueError("source mask is required when only news_embeddings are provided")
            source_mask = torch.zeros(num_nodes, 1, device=device)
        if source_mask.dim() == 1:
            source_mask = source_mask.unsqueeze(-1)
        source_mask = source_mask.to(device).float()
        news_feature = news_feature[local_idx]
        if news_feature.dim() == 1:
            news_feature = news_feature.unsqueeze(0)
        news_feature = news_feature.to(device).float()
        feature = _add_node_feature_part(
            feature, source_mask * news_feature, "news_embeddings")

    profile_feature = node_features.get("profile_embeddings")
    if profile_feature is not None:
        profile_feature = profile_feature.to(device).float()
        feature = _add_node_feature_part(feature, profile_feature, "profile_embeddings")

    return feature


def _node_feature_for_sample0(node_features, local_idx, device, seed_mask=None):
    return _node_feature_for_sample(node_features, local_idx, device, seed_mask=seed_mask)



def _move_node_features_to_device(node_features, device):
    if node_features is None:
        return None
    if torch.is_tensor(node_features):
        return node_features.to(device).float()
    return {
        key: value.to(device).float() if torch.is_tensor(value) else value
        for key, value in node_features.items()
    }


def _final_observation(influ_mat, device):
    if influ_mat.shape[1] < 2:
        raise ValueError("Diffusion samples must contain source and final infection columns.")
    return influ_mat[:, 1:2].to(device).float()




class SLCVAE_model(nn.Module):


    def __init__(self, cvae: nn.Module, gnn: nn.Module):

        super(SLCVAE_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cvae = cvae.to(self.device)
        self.gnn = gnn.to(self.device)
        self.reg_params = list(
            filter(
                lambda x: x.requires_grad,
                self.gnn.parameters()))
        
    def forward(self, seed_vec, content_feature, influ_all, train_mode):
      
        seed_hat, mean, log_var = self.cvae(seed_vec, content_feature, influ_all[:,-1].unsqueeze(-1), train_mode)
        
        if train_mode:
            # Ensure values of seed_hat are within range [0, 1]
            seed_hat.clamp(0, 1)
            # Pass seed_hat through GNN and perform propagation
           
            predictions, y = self.gnn(seed_hat, content_feature, influ_all, train_mode)
        else:
            if influ_all is None:
                predictions = None
                y = None
                return seed_hat, mean, log_var, predictions, y
            
            # Ensure values of seed_vec are within range [0, 1]
            seed_vec.clamp(0, 1)
          
            predictions, y = self.gnn(seed_vec, content_feature, influ_all, train_mode)

       
        return seed_hat, mean, log_var, predictions, y


    def train_loss(self, x, x_hat, mean, log_var, y, y_hat):

        forward_loss = F.mse_loss(y_hat, y)    
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) 
        #total_loss = forward_loss + reproduction_loss * 10 + KLD * 10
        total_loss = forward_loss + reproduction_loss + KLD 

        return total_loss

    def infer_loss(self, y_true, y_hat, x_hat, train_pred):
        
        epsilon =1e-8
        BN = nn.BatchNorm1d(1, affine=False).to(self.device)
        y_hat = y_hat.to(self.device)
        y_true = y_true.to(self.device)
        forward_loss = F.mse_loss(y_hat, y_true)
        log_pmf = []
        x_hat = [x_hat]
        for pred in train_pred:
            log_lh = torch.zeros(1).to(self.device)
            for i, x_i in enumerate(x_hat[0]):
                temp = x_i * \
                    torch.log(pred[i]+epsilon) + (1 - x_i) * torch.log(1 - pred[i]+epsilon).to(torch.double)
                temp = temp.to(self.device)
                log_lh += temp
            log_pmf.append(log_lh)
        # for pred in train_pred:
        #     log_lh = (x_hat * torch.log(pred + epsilon) +
        #             (1 - x_hat) * torch.log(1 - pred + epsilon)).sum(dim=1)
        #     log_pmf.append(log_lh)

        log_pmf = torch.stack(log_pmf)
        log_pmf = BN(log_pmf.float())

        pmf_max = torch.max(log_pmf)

        pdf_sum = pmf_max + torch.logsumexp(log_pmf - pmf_max, dim=0)

        total_loss = forward_loss*100 - pdf_sum

        return total_loss
    
    def infer_loss_test(self, y_true, y_hat):
        """
        Compute inference loss.

        Args:

        - y_true (torch.Tensor): True label tensor.

        - y_hat (torch.Tensor): Predicted label tensor.

     
        """
        
        
        y_hat = y_hat.to(self.device)
        y_true = y_true.to(self.device)
        forward_loss = F.mse_loss(y_hat, y_true)

        return forward_loss


class SLCVAE:

    def __init__(self):
        a=0
 

    def train(
            self,
            adj,
            train_dataset,
            node_features=None,
            num_thres=10,
            lr=1e-3,
            weight_decay=1e-4,
            num_epoch=100,
            print_epoch=1,
            random_seed=0,
            slcvae_model_reload=None):

        print("lr: ",lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_node = adj.shape[0]
        train_num = len(train_dataset)
        torch.manual_seed(random_seed)
        node_features = _move_node_features_to_device(node_features, self.device)
        content_dim = _node_feature_dim(node_features)
        infection_feat_dim = int(train_dataset[0].shape[1] - 1)
        cvae = CVAE(
            adj,
            input_dim=1,
            content_dim=content_dim,
            infection_feat_dim=1,
            cond_dim=64).to(self.device)
        
        gnn = GNN(adj_matrix=adj, input_dim= 1 + content_dim).to(self.device)

        slcvae_model = SLCVAE_model(cvae, gnn).to(self.device)

        if slcvae_model_reload is not None:
            slcvae_model = slcvae_model_reload.to(self.device)
            for param in slcvae_model.parameters():
                param.requires_grad = True

        optimizer = Adam(slcvae_model.parameters(), lr=lr)
        #print(slcvae_model.gnn.thres)
        # Train SLCVAE
        print("train SLCVAE:")
        slcvae_model.train()
        for epoch in range(num_epoch):
            overall_loss = 0
            for i, influ_mat in enumerate(train_dataset):   
                seed_vec = influ_mat[:, 0].to(self.device)
                influ_all = _final_observation(influ_mat, self.device)
                seed_vec = seed_vec.unsqueeze(-1).float()
                content_feature = _node_feature_for_sample(node_features, i, self.device, seed_mask=seed_vec)
                optimizer.zero_grad()
                seed_vec_hat, mean, log_var, influ_vec_hat, influ_true = slcvae_model(
                    seed_vec, content_feature, influ_all, True)
                loss = slcvae_model.train_loss(
                    seed_vec, seed_vec_hat, mean, log_var, influ_true, influ_vec_hat)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()

            average_loss = overall_loss / train_num

            if epoch % print_epoch == 0:
                f = _open_log("a")
                f.write(f"Epoch [{epoch}/{num_epoch}], loss = {average_loss:.3f}\n")
                f.close()
                print(f"Epoch [{epoch}/{num_epoch}], loss = {average_loss:.3f}")

        # Evaluation
        print("infer seed from training set:")
        f = _open_log("a+")
        f.write("infer seed from training set:\n")
        f.close()

        slcvae_model.eval()
        for param in slcvae_model.parameters():
            param.requires_grad = False


        seed_infer = []
        for i, influ_mat in enumerate(train_dataset):
            #seed_vec = influ_mat[:, 0].unsqueeze(-1).float().to(self.device)
            seed_vec = torch.zeros(influ_mat.shape[0], 1, device=self.device)
            influ_all = _final_observation(influ_mat, self.device)
            content_feature = _node_feature_for_sample(node_features, i, self.device, seed_mask=seed_vec)
            seed_vec_hat = slcvae_model.cvae(seed_vec, content_feature, influ_all[:,-1].unsqueeze(-1), train_mode=False)[0].squeeze(-1)   
            seed_infer.append(seed_vec_hat)

        for seed in seed_infer:
            seed.requires_grad = True
       

        optimizer = Adam(seed_infer, lr=lr, weight_decay=weight_decay)
        

        infer_epoch = 0    
        
        for epoch in range(infer_epoch):
            print(epoch)
            overall_loss = 0
            for i, influ_mat in enumerate(train_dataset):  
                #print(i)
                #if i%200 !=0:
                #    continue
                influ_all = _final_observation(influ_mat, self.device)
                content_feature = _node_feature_for_sample(
                    node_features, i, self.device, seed_mask=seed_infer[i].clamp(0, 1))
                # influ_vec = influ_mat[:, 1]
                # influ_vec = influ_vec.unsqueeze(-1).float()
                optimizer.zero_grad()

                seed_vec_hat, _, _, influ_vec_hat, influ_true = slcvae_model(
                    seed_infer[i], content_feature, influ_all, False)
                loss = slcvae_model.infer_loss(
                    influ_true, influ_vec_hat, seed_vec_hat, seed_vae_train)
                #print(loss)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()
            
            average_loss = overall_loss / train_num   
            # if epoch % 1 == 0:
            #     f=open("log.txt", "a")
            #     f.write(f"Epoch [{epoch}/{infer_epoch}], obj = {average_loss:.4f}\n")
            #     f.close()
            #     print(f"Epoch [{epoch}/{infer_epoch}], obj = {average_loss:.4f}")


            #print(seed_infer)

            # average_loss = overall_loss / train_num


        train_auc = 0
        pred_min = 9999
        pred_max = -9999
        for i, influ_mat in enumerate(train_dataset):

            seed_vec = influ_mat[:, 0]
            seed_vec = seed_vec.squeeze(-1).cpu().detach().numpy()
            seed_pred = seed_infer[i].cpu().detach().numpy().reshape(-1)
            pred_min = min(pred_min,seed_pred.min())
            if pred_min < 0 :
                pred_min = 0
            #print(pred_min)
            pred_max = max(pred_max,seed_pred.max())
           # train_auc += roc_auc_score(seed_vec, seed_pred)
        #train_auc = train_auc / train_num

        opt_f1 = -1
        opt_thres = -1
        opt_pr = -1
        thres_list = np.linspace(pred_min, pred_max, num=num_thres+2)[1:-1].tolist()
        for thres in thres_list:
            train_f1 = 0
            train_pr = 0
            train_re = 0
            for i, influ_mat in enumerate(train_dataset):
                seed_vec = influ_mat[:, 0]
                seed_vec = seed_vec.squeeze(-1).cpu().detach().numpy()
                seed_pred = seed_infer[i].cpu().detach().numpy().reshape(-1)
                train_f1 += f1_score(seed_vec, seed_pred >=
                                     thres, zero_division=0)
                train_pr += precision_score(seed_vec,
                                            seed_pred >= thres,
                                            zero_division=0)
                train_re += recall_score(seed_vec, seed_pred >=
                                            thres, zero_division=0)
            train_f1 = train_f1 / train_num
            train_pr = train_pr / train_num
            train_re = train_re / train_num
            f = _open_log("a")
            f.write(f"thres = {thres:.3f}, train_f1 = {train_f1:.3f}, train_pr = {train_pr:.3f}, train_re = {train_re:.3f}\n")
            f.close()
            print(f"thres = {thres:.3f}, train_f1 = {train_f1:.3f}, train_pr = {train_pr:.3f}, train_re = {train_re:.3f}")
            if train_f1 > opt_f1:
                opt_f1 = train_f1
                opt_thres = thres

        pred = np.zeros((num_node, train_num))

        for i in range(train_num):
            pred[:, i] = seed_infer[i].squeeze(-1).cpu().detach().numpy()
        
        seed_vae_train = None
        return slcvae_model, seed_vae_train, opt_thres, train_auc, opt_f1, pred

    def infer(
            self,
            test_dataset,
            slcvae_model,
            seed_vae_train,
            adj,
            node_features=None,
            thres=0.091,
            lr=0.0001,
            num_epoch=10,
            print_epoch=1):

        print("lr: ", lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node_features = _move_node_features_to_device(node_features, self.device)
        slcvae_model = slcvae_model.to(self.device)
        
        # if test_dataset.dim() == 2:
        #     test_dataset = test_dataset.unsqueeze(0)
        
        test_num = len(test_dataset)
        slcvae_model.eval()
        for param in slcvae_model.parameters():
            param.requires_grad = False
        # thres = thres + random.uniform(-0.01, 0.05)
        # print("thres:",thres)

        seed_infer = []

        for i, influ_mat in enumerate(test_dataset):
            seed_vec = torch.zeros(influ_mat.shape[0], 1, device=self.device)
            influ_all = _final_observation(influ_mat, self.device)
            content_feature = _node_feature_for_sample(node_features, i, self.device, seed_mask=seed_vec)
            seed_vec_hat = slcvae_model.cvae(seed_vec, content_feature, influ_all[:,-1].unsqueeze(-1), train_mode=False)[0]  
            seed_infer.append(seed_vec_hat)
        for seed in seed_infer:
            seed.requires_grad = True
         
        optimizer = Adam(seed_infer, lr=lr)
      
        #print(slcvae_model.gnn.thres)
        print("infer seed from test set:")

        opt_f1 = -1
        pred = np.zeros((len(seed_infer[0]), test_num))
       
        for epoch in range(num_epoch):
            overall_loss = 0
            for i, influ_mat in enumerate(test_dataset):
                influ_all = _final_observation(influ_mat, self.device)
                content_feature = _node_feature_for_sample(
                    node_features, i, self.device, seed_mask=seed_infer[i].clamp(0, 1))
                optimizer.zero_grad()
                
                seed_vec_hat, _, _, influ_vec_hat, y = slcvae_model(
                    seed_infer[i], content_feature, influ_all, False)
                # loss = slcvae_model.infer_loss_test(
                #     y, influ_vec_hat, seed_vec_hat, seed_vae_train)  

                loss = slcvae_model.infer_loss_test(y, influ_vec_hat)
                overall_loss += loss.item()

                average_loss = overall_loss / test_num

                loss.backward()
                optimizer.step()

                if i == 0:
                    # y_true = y[0:,:] # 0 3 6 9
                    # pred_y = influ_vec_hat[0:,:] #3 3 6 9
                    y_true = y[0:] # 0 3 6 9
                    pred_y = influ_vec_hat[0:] #3 3 6 9

            if epoch % print_epoch == 0:
                test_acc = 0
                test_pr = 0
                test_re = 0
                test_f1 = 0
                test_auc = 0
                valid_auc_count = 0

                preds = []
            
                for i, influ_mat in enumerate(test_dataset):
                    seed_vec = influ_mat[:, 0]
                    seed_vec = seed_vec.squeeze(-1).cpu().detach().numpy()
                    seed_pred = seed_infer[i].cpu().detach().numpy().reshape(-1)
                   
                        
                    a=seed_pred>=thres
                    test_acc += accuracy_score(seed_vec, seed_pred >= thres)
                    test_pr += precision_score(seed_vec,
                                            seed_pred >= thres,
                                            zero_division=0)
                    test_re += recall_score(seed_vec, seed_pred >=
                                            thres, zero_division=0)
                    test_f1 += f1_score(seed_vec, seed_pred >= thres, zero_division=0)
                    if np.unique(seed_vec).size > 1:
                        test_auc += roc_auc_score(seed_vec, seed_pred)
                        valid_auc_count += 1
                   
                    preds.append(a)

                test_acc = test_acc / test_num
                test_pr = test_pr / test_num
                test_re = test_re / test_num
                test_f1 = test_f1 / test_num
                test_auc = test_auc / valid_auc_count if valid_auc_count > 0 else np.nan

                metric = Metric(test_acc, test_pr, test_re, test_f1, test_auc)
                #return metric
                #print(f"Epoch [{epoch}/{num_epoch}], obj = {average_loss:.4f}")
                f = _open_log("a")
                f.write(f"Epoch [{epoch}/{num_epoch}], obj = {average_loss:.4f}, test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}\n")
                f.close()
                print(f"Epoch [{epoch}/{num_epoch}], obj = {average_loss:.4f}, test acc: {metric.acc:.3f}, test pr: {metric.pr:.3f}, test re: {metric.re:.3f}, test f1: {metric.f1:.3f}")
                if test_f1 > opt_f1:
                    opt_f1 = test_f1
                    opt_metric = metric
                    opt_y_true = y_true
                    opt_pred_y = pred_y
                    for i in range(test_num):
                        pred[:, i] = seed_infer[i].squeeze(-1).cpu().detach().numpy()
                    
        return opt_metric, pred, opt_y_true, opt_pred_y, preds
