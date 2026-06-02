import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    """
    Encoder module for a variational autoencoder (VAE).

    Attributes:

        - input_dim (int): Dimension of the input.

        - hidden_dim (int): Dimension of the hidden layer.

        - latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        """
        Initialize the Encoder.

        Args:

            - input_dim (int): Dimension of the input.

            - hidden_dim (int): Dimension of the hidden layer.

            - latent_dim (int): Dimension of the latent space.
        """
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of the Encoder.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:

            - mean (torch.Tensor): The mean of the latent space.

            - log_var (torch.Tensor): The log variance of the latent space.

        """
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        mean = self.mean(x)
        log_var = self.var(x)
        return mean, log_var


class Decoder(nn.Module):
    """
    Decoder module for a variational autoencoder (cVAE).

    Attributes:

        - output_dim (int): Dimension of the output.

        - hidden_dim (int): Dimension of the hidden layer.

        - latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        """
        Initialize the Decoder.

        Args:

            - output_dim (int): Dimension of the output.

            - hidden_dim (int): Dimension of the hidden layer.

            - latent_dim (int): Dimension of the latent space.
        """
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of the Decoder.

        Args:

            - x (torch.Tensor): Input tensor.

        Returns:

            - x_hat (torch.Tensor): Decoded output tensor.
        """
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))

        x_hat = torch.sigmoid(self.output(x))
        return x_hat



class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with infection topology as condition.

    Attributes:
        input_dim (int): Dimension of the input.
        cond_dim (int): Dimension of the condition (infection topology).
        hidden_dim (int): Dimension of the hidden layer.
        latent_dim (int): Dimension of the latent space.
    """

    def __init__(
            self,
            adj_matrix,
            input_dim=1,
            content_dim=1,
            infection_feat_dim=1,
            cond_dim=64,
            hidden_dim=512,
            latent_dim=256):
        super(CVAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cond_dim = cond_dim
        self.content_dim = content_dim
        node_input_dim = input_dim + content_dim
    
        # Encoder: source probability + content/profile match + infection condition.
        self.encoder = nn.Sequential(
            nn.Linear(node_input_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

     
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # Decoder keeps content/profile match visible during source reconstruction.
        # self.decoder = nn.Sequential(
        #     nn.Linear(2 + cond_dim + content_dim, latent_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(hidden_dim, 1),
        #     nn.Sigmoid()
        # )
        self.decoder = nn.Sequential(
            nn.Linear(2 + cond_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.linear_cond = nn.Linear(1, cond_dim) 

        self.gcn_encoder = GNNEncoder(adj_matrix, node_feat_dim=infection_feat_dim, embed_dim=cond_dim)

    def encode(self, x, cond):
       
        x_cond = torch.cat([x, cond], dim=-1)  # [..., input_dim + cond_dim]
        hidden = self.encoder(x_cond)
        mean, logvar = self.mean_layer(hidden), self.logvar_layer(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
     
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, cond):
       
        z_cond = torch.cat([z, cond], dim=-1)  # [..., 2 + cond_dim]
        return self.decoder(z_cond)

    def _prepare_content_feature(self, content_feature, num_nodes):
        if content_feature is None:
            return torch.zeros(num_nodes, self.content_dim, device=self.device)
        content_feature = content_feature.to(self.device).float()
        if content_feature.dim() == 1:
            content_feature = content_feature.unsqueeze(-1)
        return content_feature

    def forward(self, x, content_feature, y, train_mode):

        x = x.to(self.device).float()
        y = y.to(self.device).float()
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        content_feature = self._prepare_content_feature(content_feature, x.shape[0])
        cond = self.gcn_encoder(y)
        x_with_content = torch.cat([x, content_feature], dim=-1)

        #print("cond shape:", y.shape)  
        #cond = self.linear_cond(y)
        
        if train_mode:
            mean, logvar = self.encode(x_with_content, cond)
            z = self.reparameterize(mean, logvar)
        else:
            mean, logvar = None, None
            z = torch.randn(x.shape[0],2).to(self.device)
    

        x_hat = self.decode(z, cond)
        return x_hat, mean, logvar



class GCNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).

    Attributes:
        - in_features (int): Number of input features for each node.

        - out_features (int): Number of output features for each node.

        - bias (bool): Whether to include a bias term in the layer.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize a GCN layer.

        Args:
            - in_features (int): Number of input features for each node.

            - out_features (int): Number of output features for each node.

            - bias (bool): Whether to include a bias term in the layer.
        """
        super(GCNLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define a linear transformation for the layer
        self.linear = nn.Linear(in_features, out_features, bias=bias).to(self.device)

    def forward(self, x, adj):
        """
        Forward pass of the GCN layer.

        Args:
            - x (torch.Tensor): Input feature matrix of shape (num_nodes, in_features).

            - adj (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).

        Returns:
            - x(torch.Tensor): Output feature matrix of shape (num_nodes, out_features).
        """
        # Perform the graph convolution operation
        if getattr(adj, "is_sparse", False):
            x = torch.sparse.mm(adj, x)
        else:
            x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) model using GCN layers.

    Attributes:
        - adj_matrix (torch.Tensor): Adjacency matrix representing graph connectivity.

        - input_dim (int): Dimension of the input.

        - hiddenunits (List[int]): List of hidden units for each layer.

        - num_classes (int): Number of output classes.

        - bias (bool): Whether to include bias in linear layers.

        - drop_prob (float): Dropout probability.
    """

    def __init__(
            self,
            adj_matrix,
            input_dim=1,
            hiddenunits=[
                512,
                64],
            out_dim=1,
            bias=True,
            drop_prob=0.5):
        """
        Initialize the GNN model.

        Args:
            - adj_matrix (torch.Tensor): Adjacency matrix representing graph connectivity.

            - input_dim (int): Dimension of the input.

            - hiddenunits (List[int]): List of hidden units for each layer.

            - out_dim (int): Dimension of the output.

            - bias (bool): Whether to include bias in linear layers.

            - drop_prob (float): Dropout probability.
        """
        super(GNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim

        if sp.isspmatrix(adj_matrix):
            adj_coo = adj_matrix.tocoo()
            indices = torch.vstack((
                torch.from_numpy(adj_coo.row).long(),
                torch.from_numpy(adj_coo.col).long()))
            values = torch.from_numpy(adj_coo.data).float()
            adj_tensor = torch.sparse_coo_tensor(
                indices,
                values,
                torch.Size(adj_coo.shape),
                dtype=torch.float32).coalesce()
        else:
            adj_tensor = torch.as_tensor(adj_matrix, dtype=torch.float32)

        self.register_buffer("adj_matrix", adj_tensor.to(self.device))

        # Define GCN layers
        gcn_layers = [GCNLayer(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            gcn_layers.append(GCNLayer(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        gcn_layers.append(GCNLayer(hiddenunits[-1], out_dim, bias=bias))

        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Define dropout layer
        self.dropout = nn.Dropout(drop_prob).to(self.device) if drop_prob > 0 else lambda x: x

        # Activation function
        self.act_fn = nn.ReLU().to(self.device)

    def forward(self, seed_vec, content_feature=None, influ_all=None, train_mode=None):
        """
        Forward pass of the GNN.

        Args:
            - seed_vec (torch.Tensor): Input seed vector.

        Returns:
            - x (torch.Tensor): Predicted output.
        """
        seed_vec = seed_vec.to(self.device).float()
        if seed_vec.dim() == 1:
            seed_vec = seed_vec.unsqueeze(-1)
        if content_feature is None:
            extra_dim = self.input_dim - seed_vec.shape[-1]
            if extra_dim > 0:
                content_feature = torch.zeros(seed_vec.shape[0], extra_dim, device=self.device)
        else:
            content_feature = content_feature.to(self.device).float()
            if content_feature.dim() == 1:
                content_feature = content_feature.unsqueeze(-1)
        x = seed_vec if content_feature is None else torch.cat([seed_vec, content_feature], dim=-1)
        # x = content_feature
        # Apply each GCN layer
        for layer in self.gcn_layers[:-1]:
            x = self.act_fn(layer(x, self.adj_matrix))
            x = self.dropout(x)
        
        # Final layer with no activation function
        res = torch.sigmoid(self.gcn_layers[-1](x, self.adj_matrix))

        return res, influ_all[:, -1:]
    

    def loss(self, y, y_hat):
        """
        Calculate loss.

        Args:
            - y (torch.Tensor): Ground truth.
            - y_hat (torch.Tensor): Predicted output.

        Returns:
            - forward_loss (torch.Tensor): Forward loss.
        """
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss
    

class GNNEncoder(torch.nn.Module):
    def __init__(self, adj_matrix, node_feat_dim=1, embed_dim=64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adj_matrix = adj_matrix
        # Convert sparse matrix to dense if needed
        #if sp.isspmatrix(adj_matrix):
        #    adj_torch = adj_matrix.toarray()
        #self.adj_torch = nn.Parameter(torch.Tensor(adj_torch).to(self.device),requires_grad=False)
        edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
        self.register_buffer("edge_index", edge_index.to(self.device))
        self.register_buffer("edge_weight", edge_weight.to(self.device).float())
        self.conv1 = GCNConv(node_feat_dim, 64)
        self.conv2 = GCNConv(64, embed_dim)
        self.predictor = torch.nn.Linear(32, 1)

    def forward(self, x):

        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.conv1(x, self.edge_index,edge_weight=self.edge_weight).relu().to(torch.float32)
        x = self.conv2(x, self.edge_index,edge_weight=self.edge_weight).relu().to(torch.float32)
        #x = torch.sigmoid(self.predictor(x))
    
        return x
