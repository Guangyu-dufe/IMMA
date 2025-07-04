import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv, ChebGraphConv
from model.team_model import make_model
import sys
from torch_geometric.utils import dense_to_sparse
from model.dlf_model import DSTG

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class MLP_Model(nn.Module):
    """Some Information about MLP"""
    def __init__(self, args):
        super(MLP_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).transpose(1, 2).unsqueeze(-1)
        
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden).squeeze(-1).reshape(1, 2)
        x = prediction.reshape(-1, 12)
        return x



class LSTM_Model(nn.Module):
    """Some Information about LSTM"""
    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).unsqueeze(-1).transpose(1, 2).transpose(1, 3)   # [bs, t, n, f]
        b, f, n, t = x.shape

        x = x.transpose(1,2).reshape(b*n, f, 1, t)  # (b, f, n, t) -> (b, n, f, t) -> (b * n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)  # (b * n, f, 1, t) -> (b * n, init_dim, 1, t) -> (b * n, init_dim, t) -> (b * n, t, init_dim)

        out, _ = self.lstm(x)  # (b * n, t, hidden_dim) -> (b * n, t, hidden_dim)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b*n, t)
        return x


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=10):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_a = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_dim, r)))
        self.lora_b = nn.Parameter(torch.zeros(r, out_dim))
        self.scaling = 1 / (r * in_dim)

    def forward(self, x):
        return x + self.scaling * torch.matmul(torch.matmul(x, self.lora_a.to(x.device)), self.lora_b.to(x.device))
    

class STLora_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(STLora_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        self.lora_layers = nn.ModuleList()  # 存放LoRA层的列表
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    
    def add_lora_layer(self):
        in_dim = self.args.gcn["hidden_channel"]
        out_dim = self.args.gcn["hidden_channel"]
        lora_layer = LoRALayer(in_dim, out_dim)
        self.lora_layers.append(lora_layer)
        self.freeze_lora_layers()  # 冻结现有的LoRA层
    
    def freeze_lora_layers(self):
        for lora_layer in self.lora_layers[:-1]:  # 冻结除了最后一个之外的所有LoRA层
            for param in lora_layer.parameters():
                param.requires_grad = False

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        x = self.tcn1(x)                                           # [bs * N, 1, feature]
        
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x


class EAC_Model(nn.Module):
    """Some Information about EAC_Model"""
    def __init__(self, args):
        super(EAC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Set a low rank value
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], 
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        # Initialize subspace and adjust matrix
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        B, N, T = x.shape
        
        # Compute adaptive parameters using low-rank matrices
        adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            
            self.num_nodes = new_num_nodes




class TrafficStream_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(TrafficStream_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x




class STKEC_Model(nn.Module):
    """Some Information about STKEC_Model"""
    def __init__(self, args):
        super(STKEC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.ReLU()

        self.memory=nn.Parameter(torch.zeros(size=(args.cluster, args.gcn["out_channel"]), requires_grad=True))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        
    def forward(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]
        scores = F.softmax(attention, dim=1)                       # [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x, scores
    
    def feature(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        return x


class Universal_Model(nn.Module):
    def __init__(self, args):
        super(Universal_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.use_eac = args.use_eac
        
        # Initialize GCN layers based on spectral (sp) or spatial (st) options
        if args.gcn_type == 'st':
            self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
            self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["in_channel"], bias=True, gcn=False)
        elif args.gcn_type == 'sp':
            self.gcn1 = ChebGraphConv(args.gcn["in_channel"], args.gcn["hidden_channel"])
            self.gcn2 = ChebGraphConv(args.gcn["hidden_channel"], args.gcn["in_channel"])
        
        # Select TCN type based on args
        if args.tcn_type == 'conv':
            self.tcn = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], 
                                kernel_size=args.tcn["kernel_size"],
                                dilation=args.tcn["dilation"],
                                padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2))
        elif args.tcn_type == 'rec':
            self.tcn = nn.LSTM(input_size=args.gcn["hidden_channel"], hidden_size=args.gcn["hidden_channel"], batch_first=True)
        elif args.tcn_type == 'attn':
            self.tcn = nn.MultiheadAttention(embed_dim=args.gcn["hidden_channel"], num_heads=4)
        
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        if self.use_eac:
            self.rank = args.rank  # 设定低秩的值
            self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
            self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
            self.year = args.year
            self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # [bs, N, feature]
        
        B, N, T = x.shape
        
        if self.use_eac:
            adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
            x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        # Apply the selected GCN layers
        x = F.relu(self.gcn1(x, adj))  # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # [bs * N, 1, feature]
        
        # Apply the selected TCN method
        if self.args.tcn_type == 'conv':
            x = self.tcn(x)  # temporal convolution
        elif self.args.tcn_type == 'rec':
            # x = x.reshape((-1, self.args.gcn["hidden_channel"])).unsqueeze(dim=-1)
            # out, _ = self.tcn(x)
            # x = out.reshape((-1, 1, self.args.gcn["hidden_channel"]))
            x = x.reshape(B, N, self.args.gcn["hidden_channel"])
            x, _ = self.tcn(x)
            x = x.reshape(B*N, 1, self.args.gcn["hidden_channel"])
        elif self.args.tcn_type == 'attn':
            x = x.reshape(B, N, self.args.gcn["hidden_channel"])
            x, _ = self.tcn(x, x, x)  # Multihead attention
            x = x.reshape(B*N, 1, self.args.gcn["hidden_channel"])
        
        
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # [bs, N, feature]
        x = self.gcn2(x, adj)  # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            
            self.num_nodes = new_num_nodes


class PECPM_Model(nn.Module):
    """PECPM Model with Pattern Expansion and Consolidation based on Pattern Matching"""
    def __init__(self, args):
        super(PECPM_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        # Pattern memory bank for PECPM
        self.memory = nn.Parameter(torch.FloatTensor(args.cluster, args.gcn["out_channel"]), requires_grad=True)
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        
    def forward(self, data, adj):
        data.x = torch.nan_to_num(data.x, nan=0.0)
        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   
        res_x = data.x.reshape(-1, self.args.gcn["out_channel"])

        x = F.relu(self.gcn1(x, adj))                              
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    
        x = self.tcn1(x)                                           
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))   
        x = self.gcn2(x, adj)                                      
        x = x.reshape((-1, self.args.gcn["out_channel"]))        

        # Pattern matching attention mechanism
        attention = torch.matmul(x, self.memory.transpose(0, 1))   
        attention = F.softmax(attention, dim=1)                   
        z = torch.matmul(attention, self.memory)                  

        # Residual connection and final prediction
        x = x + res_x + z                                     
        x = self.fc(self.activation(x))                          
        x = F.dropout(x, p=self.dropout, training=self.training)  

        return x, attention

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  
        x = F.relu(self.gcn1(x, adj))                             
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    
        x = self.tcn1(x)                                          
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    
        x = self.gcn2(x, adj)                                      
        x = x.reshape((-1, self.args.gcn["out_channel"]))         
        x = x + data.x
        return x
    

class TEAM_Model(nn.Module):
    """TEAM Model wrapper for integration with the main framework"""
    def __init__(self, args):
        super(TEAM_Model, self).__init__()
        self.args = args
        
        self.team_model = make_model(
            DEVICE=args.device,
            nb_block=getattr(args, 'nb_block', 2),
            in_channels=1,
            K=getattr(args, 'K', 3),
            nb_chev_filter=getattr(args, 'nb_chev_filter', 64),
            nb_time_filter=getattr(args, 'nb_time_filter', 64),
            time_strides=getattr(args, 'time_strides', 1),
            num_for_predict=args.y_len,
            len_input=args.x_len
        )
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def _adj_to_edge_index(self, adj):
        """将邻接矩阵转换为 PyTorch Geometric 接受的边索引格式."""
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj)

        if not isinstance(adj, torch.Tensor):
            raise TypeError(f"输入必须是 torch.Tensor 或 np.ndarray, 但收到 {type(adj)}")

        if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"邻接矩阵必须是方的 2D 张量, 但收到形状 {adj.shape}")

        edge_index, _ = dense_to_sparse(adj)

        try:
            device = self.args.device
        except AttributeError:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return edge_index.to(device)
    
    def forward(self, data, adj):
        """Forward pass adapted for the main framework"""
        # 确保所有输入都在正确的设备上
        if hasattr(data, 'x'):
            data.x = data.x.to(self.args.device)
        if isinstance(adj, torch.Tensor):
            adj = adj.to(self.args.device)
            
        edge_index = self._adj_to_edge_index(adj)
        
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        T = data.x.shape[1] 
        
        x = data.x.reshape(batch_size, N, T)  # [batch_size, N, T]
        x = x.unsqueeze(-2)  # [batch_size, N, 1, T]

        output = self.team_model(x, edge_index)  # [batch_size, N, output_dim]
        output = output.reshape(-1, output.shape[-1])  # [batch_size * N, output_dim]
        
        return output
    
    def feature(self, data, adj):
        if hasattr(data, 'x'):
            data.x = data.x.to(self.args.device)
        if isinstance(adj, torch.Tensor):
            adj = adj.to(self.args.device)
            
        edge_index = self._adj_to_edge_index(adj)
        
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        T = data.x.shape[1] 
        
        x = data.x.reshape(batch_size, N, T)  # [batch_size, N, T]
        x = x.unsqueeze(-2)  # [batch_size, N, 1, T]
        
        features = self.team_model.feature(x, edge_index)
        
        return features.reshape(-1, features.shape[-1])


class DLF_Model(nn.Module):
    """TEAM Model wrapper for integration with the main framework"""
    def __init__(self, args):
        super(DLF_Model, self).__init__()
        self.args = args
        self.dlf_model = DSTG(
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
            dilation_channels=args.dilation_channels,
            skip_channels=args.skip_channels,
            end_channels=args.end_channels,
            supports_len=args.supports_len,
            kernel_size=args.kernel_size,
            blocks=args.blocks,
            layers=args.layers,
            device=args.device,
            input_dim = 1,
            output_dim = 1,

        )
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        if hasattr(data, 'x'):
            data.x = data.x.to(self.args.device)
        if isinstance(adj, torch.Tensor):
            adj = adj.to(self.args.device)
            
        
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        T = data.x.shape[1] 
        
        x = data.x.reshape(batch_size, N, T)  # [batch_size, N, T]
        x = x.unsqueeze(-2).permute(0,3,1,2)  # [batch_size, N, 1, T]

        output = self.dlf_model(x, [adj]).permute(0,2,1,3).squeeze(-1) # [batch_size,T, N, 1]
        output = output.reshape(-1, output.shape[-1])  # [batch_size * N, output_dim]
        
        return output
    
    def feature(self, data, adj):
        if hasattr(data, 'x'):
            data.x = data.x.to(self.args.device)
        if isinstance(adj, torch.Tensor):
            adj = adj.to(self.args.device)
            
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        T = data.x.shape[1] 
        
        x = data.x.reshape(batch_size, N, T)  # [batch_size, N, T]
        x = x.unsqueeze(-2).permute(0,3,1,2)  # [batch_size, N, 1, T]

        output = self.dlf_model(x, [adj]).permute(0,2,1,3).squeeze(-1) # [batch_size,T, N, 1]
        output = output.reshape(-1, output.shape[-1])  # [batch_size * N, output_dim]
        
        return output

