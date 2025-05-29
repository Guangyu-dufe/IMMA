# -*- coding:utf-8 -*-
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ChebConv, TransformerConv
from torch_geometric.nn import GATConv


class Spatial_Attention_layer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, DEVICE, in_features, out_features, alpha, concat=True):
        super(Spatial_Attention_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features).to(DEVICE))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1).to(DEVICE))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)       
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = torch.matmul(h, self.W)                            # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)       # e (b, N, N)
        attention = torch.sigmoid(e)
        attention = F.softmax(attention, dim=1)             
        h_prime = torch.matmul(attention, Wh)               
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)




class Spatial_Transformer_layer(nn.Module):
    '''
    Compute the output base on the asigned weight of Spatial Attention layer
    '''
    def __init__(self, DEVICE, in_channels, out_features, nheads = 1, alpha = 0.2):

        super(Spatial_Transformer_layer, self).__init__()
        self.D = out_features // nheads         # output of each attention head
        self.attentions = [Spatial_Attention_layer(DEVICE, in_channels, self.D, alpha) for _ in range(nheads)]      # each head has heads * (B,N,out_features // nheads)
        self.out_att = Spatial_Attention_layer(DEVICE, nheads * self.D, out_features, alpha, concat = False)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F * T)
        :return:  (batch_size, N, F * T)
        '''
        x = torch.cat([att(x) for att in self.attentions], dim=2)
        x = F.relu(self.out_att(x))    
        return x





class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, out_channels, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(DEVICE))              # ???
        self.U2 = nn.Parameter(torch.FloatTensor(out_channels).to(DEVICE))                           # ???
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 1, 2), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(E, dim=1)

        return E_normalized             # (B, T, T)



class cheb_interpolation_conv(nn.Module):
    '''
    K-order chebyshev graph convolution with interpolation (ChebNet II)
    '''

    def __init__(self, DEVICE, K, cheb_poly_xj, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_interpolation_conv, self).__init__()
        self.K = K
        self.cheb_poly_xj = cheb_poly_xj
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = DEVICE
        self.Gamma = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, cheb_polynomials):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = cheb_polynomials[k]  # (N,N)
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)    # (b, F_in, N)(N,N) -> (b, F_in, N) -> (b, N, F_in)
                lhs = torch.zeros(self.in_channels, self.out_channels).to(self.DEVICE)
                for j in range (self.K):
                    Tk_xj = self.cheb_poly_xj[j][k]    # (1,1)
                    gamma_j = self.Gamma[j]      # (F_in, F_out)
                    lhs = lhs + gamma_j * Tk_xj   # (F_in, F_out)
                output = output + rhs.matmul(lhs)   # (b, N, F_in)(F_in,F_out) -> (b, N, F_out)

            outputs.append(output.unsqueeze(-1))

        return F.relu( 2 / (self.K+1) * torch.cat(outputs, dim=-1))


class COAT_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_timesteps):
        super(COAT_block, self).__init__()
        
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.DEVICE = DEVICE
        self.num_of_timesteps = num_of_timesteps
        
        # Spatial convolution layers
        self.cheb_conv = ChebConv(in_channels, nb_chev_filter, K).to(DEVICE)
        self.gat = GATConv(nb_chev_filter, nb_chev_filter, heads=2, concat=False, dropout=0.1).to(DEVICE)
        
        # Temporal layers
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), 
                                 stride=(1, time_strides), padding=(0, 1)).to(DEVICE)
                                 
        self.temporal_attention = nn.MultiheadAttention(nb_time_filter, num_heads=4, batch_first=True).to(DEVICE)
        
        # Residual and output layers
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1),
                                     stride=(1, time_strides)).to(DEVICE)
                                     
        self.ln = nn.LayerNorm(nb_time_filter).to(DEVICE)
        
        self.backcast_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                     stride=(1, time_strides), padding=(0, 1)).to(DEVICE)
        self.forecast_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                     stride=(1, time_strides), padding=(0, 1)).to(DEVICE)

    def forward(self, x, edge_index):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        # Move inputs to correct device
        x = x.to(self.DEVICE)
        edge_index = edge_index.to(self.DEVICE)
        
        # Process each timestep with ChebConv
        spatial_outs = []
        for t in range(num_of_timesteps):
            # Extract features for current timestep
            x_t = x[:, :, :, t].reshape(-1, num_of_features)  # (batch_size * num_vertices, features)
            
            # Apply ChebConv
            out_t = self.cheb_conv(x_t, edge_index)  # (batch_size * num_vertices, nb_chev_filter)
            
            # Apply GAT
            out_t = self.gat(out_t, edge_index)  # (batch_size * num_vertices, nb_chev_filter)
            
            # Reshape back
            out_t = out_t.reshape(batch_size, num_of_vertices, self.nb_chev_filter)
            spatial_outs.append(out_t)
            
        # Stack temporal dimension
        spatial_out = torch.stack(spatial_outs, dim=-1)  # (batch_size, num_vertices, nb_chev_filter, timesteps)
        
        # Temporal Convolution
        temporal_conv = self.time_conv(spatial_out.permute(0, 2, 1, 3))  # (batch_size, nb_time_filter, num_vertices, timesteps)
        temporal_conv = temporal_conv.permute(0, 2, 1, 3)  # (batch_size, num_vertices, nb_time_filter, timesteps)
        
        # Temporal Attention
        temp_in = temporal_conv.reshape(batch_size * num_of_vertices, num_of_timesteps, self.nb_time_filter)
        temp_out, _ = self.temporal_attention(temp_in, temp_in, temp_in)
        temp_out = temp_out.reshape(batch_size, num_of_vertices, self.nb_time_filter, num_of_timesteps)
        
        # Residual Connection
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        result = self.ln(F.relu(x_residual + temp_out).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Output Projections
        backcast = self.backcast_conv(result.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        forecast = self.forecast_conv(result.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        
        return backcast.to(self.DEVICE), forecast.to(self.DEVICE)



class COAT_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 
                 time_strides, num_for_predict, len_input):
        super(COAT_submodule, self).__init__()
        
        self.BlockList = nn.ModuleList([
            COAT_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, 
                      time_strides, len_input)
        ])
        
        self.BlockList.extend([
            COAT_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                      len_input//time_strides) for _ in range(nb_block-1)
        ])
        
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict,
                                  kernel_size=(1, nb_time_filter)).to(DEVICE)
        
        self.nb_time_filter = nb_time_filter
        self.DEVICE = DEVICE

    def forward(self, x, edge_index):
        # Move input to device
        x = x.to(self.DEVICE)
        edge_index = edge_index.to(self.DEVICE)
        
        z = torch.zeros(size=(x.shape[0], x.shape[1], self.nb_time_filter, x.shape[3])).to(self.DEVICE)
        
        for block in self.BlockList:
            x_l, h = block(x, edge_index)
            x = x - x_l.to(self.DEVICE)  # Ensure same device
            z = z + h.to(self.DEVICE)  # Ensure same device
            
        output = self.final_conv(z.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output.to(self.DEVICE)  # Ensure output is on correct device
        
    def feature(self, x, edge_index):
        # Move input to device
        x = x.to(self.DEVICE)
        edge_index = edge_index.to(self.DEVICE)
        
        z = torch.zeros(size=(x.shape[0], x.shape[1], self.nb_time_filter, x.shape[3])).to(self.DEVICE)
        
        for block in self.BlockList:
            x_l, h = block(x, edge_index)
            x = x - x_l.to(self.DEVICE)  # Ensure same device
            z = z + h.to(self.DEVICE)  # Ensure same device
            
        return z.to(self.DEVICE)  # Ensure output is on correct device



def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 
               time_strides, num_for_predict, len_input):
               
    model = COAT_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter,
                          nb_time_filter, time_strides, num_for_predict, len_input)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
            
    return model

def adj_to_edge_index(adj):
    """将邻接矩阵转换为边索引格式
    Args:
        adj: torch.Tensor or np.ndarray, shape (N, N) - 邻接矩阵
    Returns:
        edge_index: torch.Tensor, shape (2, E) - 边索引
    """
    # 确保输入是PyTorch张量
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    
    # 获取非零元素的索引
    edge_index = torch.nonzero(adj).t()
    
    return edge_index

def test_model():
    """测试TEAM模型的功能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 模型参数
    nb_block = 2
    in_channels = 1
    K = 3
    nb_chev_filter = 64
    nb_time_filter = 64
    time_strides = 1
    num_for_predict = 12
    len_input = 12
    
    # 创建模型
    model = make_model(
        DEVICE=device,
        nb_block=nb_block,
        in_channels=in_channels,
        K=K,
        nb_chev_filter=nb_chev_filter,
        nb_time_filter=nb_time_filter,
        time_strides=time_strides,
        num_for_predict=num_for_predict,
        len_input=len_input
    )
    print("Model created successfully")
    
    # 生成测试数据
    batch_size = 32
    num_nodes = 100
    x = torch.randn(batch_size, num_nodes, in_channels, len_input).to(device)
    
    # 生成随机邻接矩阵并转换为边索引
    adj = torch.rand(num_nodes, num_nodes) > 0.8  # 生成稀疏邻接矩阵
    adj = adj.float()
    adj = adj + adj.t()  # 确保对称
    adj[adj > 0] = 1
    print(f"\nAdjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix density: {adj.sum() / (num_nodes * num_nodes):.3f}")
    
    edge_index = adj_to_edge_index(adj).to(device)
    print(f"Edge index shape: {edge_index.shape}")
    
    # 前向传播测试
    try:
        output = model(x, edge_index)
        print("\nForward pass successful")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print("\nForward pass failed")
        print(f"Error: {str(e)}")
    
    # 特征提取测试
    try:
        features = model.feature(x, edge_index)
        print("\nFeature extraction successful")
        print(f"Feature shape: {features.shape}")
    except Exception as e:
        print("\nFeature extraction failed")
        print(f"Error: {str(e)}")
    
    # 测试模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, x, edge_index, output

if __name__ == "__main__":
    test_model()