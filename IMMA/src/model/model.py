import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_conv import BatchGCNConv
from .GraphWaveNet import GWNET
from .dcrnn import DCRNN
from .stgcn import STGCN
from .mtgnn import MTGNN
from .STAEformer import SelfAttentionLayer
class Basic_Model_org(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model_org, self).__init__()
        self.dropout = args.dropout
        
        if args.expand:
            self.gcn1 = BatchGCNConv(args.gcn["in_channel"]*2, args.gcn["hidden_channel"], bias=True, gcn=False)
            self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
            self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
                dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        else:
            self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
            self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
            self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
                dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)

        self.activation = nn.GELU()

        self.args = args

    def forward(self, data, adj):
        N = adj.shape[0]
        if self.args.expand:
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]*2))   # [bs, N, feature]
                
            x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
            x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

            x = self.tcn1(x)                                           # [bs * N, 1, feature]

            x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
            x = self.gcn2(x, adj)                                      # [bs, N, feature]
            x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
            
            x = x + data.x[..., :self.args.gcn["in_channel"]]
                
            x = self.fc(self.activation(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return x
        else:
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
        if self.args.extra_feature:
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]*2))[:, :, :self.args.gcn["in_channel"]]
            data.x = data.x[..., :self.args.gcn["in_channel"]]
        else:
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))[:, :, :self.args.gcn["in_channel"]]   # [bs, N, feature]
        
        if self.args.expand and len(data.x.shape) == 2:
            x = torch.cat([x, data.x[..., :self.args.gcn["in_channel"]].reshape(-1, N, self.args.gcn["in_channel"])], dim=-1)
        elif self.args.expand:
            x = torch.cat([x, data.x.reshape(-1, N, self.args.gcn["in_channel"])], dim=-1)
        else:
            pass

        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
            
        return x

class STAEformer_Model(nn.Module):
    def __init__(self, args):
        super(STAEformer_Model, self).__init__()
        self.args = args
        self.x_proj = nn.Linear(1, 4)

        self.tem_attn = nn.ModuleList([SelfAttentionLayer(4*3,1024,12) for _ in range(1)])
        # self.spatial_attn = nn.ModuleList([SelfAttentionLayer(4*3,1024,12) for _ in range(12)])
        self.output_proj = nn.Linear(4*3, 1)
        # 不再预先定义每年的节点数量，而是使用动态embedding
        self.spatial_emb_dim = 4
        self.adaptive_emb_dim = 4
        
    def get_embeddings(self, N):
        """根据当前节点数量动态创建或获取embedding"""
        device = next(self.parameters()).device
        
        # 如果embedding不存在或大小不匹配，就创建新的
        if not hasattr(self, '_spatial_emb') or self._spatial_emb.size(0) != N:
            # 直接创建新的embedding，不保留之前的权重
            self._spatial_emb = nn.Parameter(torch.randn(N, self.spatial_emb_dim, device=device))
            nn.init.xavier_uniform_(self._spatial_emb)
        if not hasattr(self, '_adaptive_emb') or self._adaptive_emb.size(1) != N:
            # 直接创建新的embedding，不保留之前的权重
            self._adaptive_emb = nn.Parameter(torch.randn(12, N, self.adaptive_emb_dim, device=device))
            nn.init.xavier_uniform_(self._adaptive_emb)
        return self._spatial_emb, self._adaptive_emb

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))[:, :, :self.args.gcn["in_channel"]]   # [bs, N, feature]
        
        x = x.unsqueeze(-1) # (B, N, T, 1)
        
        # 动态获取当前节点数量对应的embedding
        spatial_emb, adaptive_embedding = self.get_embeddings(N)
        
        x = self.x_proj(x) #(B,N,T,2)
        spatial_emb = spatial_emb.expand(
                x.shape[0], 12, *spatial_emb.shape
            ).transpose(1,2) #(B,N,T,2)

        adaptive_embedding = adaptive_embedding.expand(
                 size=(x.shape[0],*adaptive_embedding.shape)
            ).transpose(1,2) #(B,N,T,2)

        x = torch.cat([x, spatial_emb, adaptive_embedding], dim=-1) #(B,N,T,6)
        # Temporal attention on T dimension

        x=x.transpose(1,2)
        for i in range(1):
            x = self.tem_attn[i](x,dim=1) #(B,T,N,D)
        x=x.transpose(1,2)
        # x = self.spatial_attn(x,dim=2)
        # x=x.transpose(1,2)
        # # Spatial attention on N dimension
        # x_spatial = x.permute(0, 2, 1, 3)  # (B, T, N, D)
        # x_spatial = x_spatial.reshape(B * T, N, D).permute(1, 0, 2)  # (N, B*T, D)
        # x_spatial_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)  # (N, B*T, D)
        # x_spatial_attn = x_spatial_attn.permute(1, 0, 2).reshape(B, T, N, D)  # (B, T, N, D)
        # x_spatial_attn = x_spatial_attn.permute(0, 2, 1, 3)  # (B, N, T, D)
        
        # Combine temporal and spatial attention
        x = self.output_proj(x).squeeze(-1)
        x = x.reshape((-1, self.args.gcn["out_channel"]))

        return x
    def feature(self, data, adj):
        N = adj.shape[0]
        if self.args.extra_feature:
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]*2))[:, :, :self.args.gcn["in_channel"]]
            data.x = data.x[..., :self.args.gcn["in_channel"]]
        else:
            x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))[:, :, :self.args.gcn["in_channel"]]   # [bs, N, feature]
        
        # x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
        x = x.unsqueeze(-1) # (B, N, T, 1)
        
        # 动态获取当前节点数量对应的embedding
        spatial_emb, adaptive_embedding = self.get_embeddings(N)
        
        x = self.x_proj(x) #(B,N,T,2)
        spatial_emb = spatial_emb.expand(
                x.shape[0], 12, *spatial_emb.shape
            ).transpose(1,2) #(B,N,T,2)

        adaptive_embedding = adaptive_embedding.expand(
                 size=(x.shape[0],*adaptive_embedding.shape)
            ).transpose(1,2) #(B,N,T,2)

        x = torch.cat([x, spatial_emb, adaptive_embedding], dim=-1) #(B,N,T,6)
        # Temporal attention on T dimension
        B, N, T, D = x.shape
        x_temp = x.permute(0, 1, 3, 2)  # (B, N, D, T)
        x_temp = x_temp.reshape(B * N, D, T).permute(2, 0, 1)  # (T, B*N, D)
        x_temp_attn, _ = self.tem_attn(x_temp, x_temp, x_temp)  # (T, B*N, D)
        x_temp_attn = x_temp_attn.permute(1, 2, 0).reshape(B, N, D, T)  # (B, N, D, T)
        x_temp_attn = x_temp_attn.permute(0, 1, 3, 2)  # (B, N, T, D)
        
        # Spatial attention on N dimension
        # x_spatial = x.permute(0, 2, 1, 3)  # (B, T, N, D)
        # x_spatial = x_spatial.reshape(B * T, N, D).permute(1, 0, 2)  # (N, B*T, D)
        # x_spatial_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)  # (N, B*T, D)
        # x_spatial_attn = x_spatial_attn.permute(1, 0, 2).reshape(B, T, N, D)  # (B, T, N, D)
        # x_spatial_attn = x_spatial_attn.permute(0, 2, 1, 3)  # (B, N, T, D)
        
        # Combine temporal and spatial attention
        x = self.output_proj(x)
        # x = x.reshape((-1, self.args.gcn["out_channel"]))
        return x

class GWNET_Model(nn.Module):
    def __init__(self, args):
        super(GWNET_Model, self).__init__()
        self.args = args
        self.gwnet = GWNET(device=args.device)
        
    def forward(self, data, adj):
         N = adj.shape[0]
         x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
         x = x.unsqueeze(1) # (B, C, N, T)
         x = self.gwnet(x,adj)
         x = x.reshape((-1, self.args.gcn["out_channel"]))
         return x
    def feature(self, data, adj):
         N = adj.shape[0]
         x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
         x = x.unsqueeze(1) # (B, C, N, T)
         x = self.gwnet(x,adj)
         x = x.reshape((-1, self.args.gcn["out_channel"]))
         return x

class STGCN_Model(nn.Module):
    def __init__(self, args):
        super(STGCN_Model, self).__init__()
        self.args = args
        self.stgcn = STGCN(device=args.device)
        
    def forward(self, data, adj):
         N = adj.shape[0]
         x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
         x = x.unsqueeze(1) # (B, C, N, T)
         x = x.permute(0, 3 , 2, 1) # (B, T, N, C)
         x = self.stgcn(x,adj)
         x = x.reshape((-1, self.args.gcn["out_channel"]))
         return x
    def feature(self, data, adj):
         N = adj.shape[0]
         x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
         x = x.unsqueeze(1) # (B, C, N, T)
         x = x.permute(0, 3 , 2, 1) # (B, T, N, C)
         x = self.stgcn(x,adj)
         x = x.reshape((-1, self.args.gcn["out_channel"]))
         return x  

    
class DCRNN_Model(nn.Module):
    def __init__(self, args):
        super(DCRNN_Model, self).__init__()
        self.args = args
        self.dcrnn =  DCRNN(
        device=args.device,
        input_dim=1,
        output_dim=1,
        horizon=12, 
    )
    def forward(self, data, adj):
         N = adj.shape[0]
         x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
         x = x.unsqueeze(1) # (B, C, N, T)
         x = self.dcrnn(x,adj)
         x = x.reshape((-1, self.args.gcn["out_channel"]))
         return x
    def feature(self, data, adj):
         N = adj.shape[0]
         x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, T]
         x = x.unsqueeze(1) # (B, C, N, T)
         x = self.dcrnn(x,adj)
         x = x.reshape((-1, self.args.gcn["out_channel"]))
         return x

class MTGNN_Model(nn.Module):
    def __init__(self, args):
        super(MTGNN_Model, self).__init__()
        self.args = args
        self.mtgnn = MTGNN(
            gcn_true=True,
            buildA_true=True,
            gcn_depth=2,
            num_nodes=8196,  # 使用配置中的节点数，默认207
            device=args.device,
            predefined_A=None,
            static_feat=None,
            dropout=args.dropout,
            subgraph_size=20,
            node_dim=8,
            dilation_exponential=1,
            conv_channels=8,
            residual_channels=8,
            skip_channels=16,
            end_channels=32,
            seq_length=args.x_len,
            in_dim=1,  # 输入特征维度
            out_dim=args.y_len,  # 输出预测长度
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True
        )
        
    def forward(self, data, adj):
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        
        # 重塑输入数据为 [batch_size, seq_length, num_nodes, in_dim]
        x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]))   # [bs, N, T]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [bs, T, N, 1]
        
        # 如果节点数与模型训练时不同，需要传递节点索引
        if N != self.mtgnn.num_nodes:
            node_idx = torch.arange(N).to(self.args.device) % self.mtgnn.num_nodes
            output = self.mtgnn(x, idx=node_idx)
        else:
            output = self.mtgnn(x)
        
        # 输出形状为 [batch_size, out_dim, num_nodes, 1]
        # 重塑为 [batch_size * num_nodes, out_dim]
        output = output.squeeze(-1).permute(0, 2, 1)  # [bs, N, out_dim]
        output = output.reshape((-1, self.args.gcn["out_channel"]))
        return output
        
    def feature(self, data, adj):
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        
        # 重塑输入数据为 [batch_size, seq_length, num_nodes, in_dim]
        x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]))   # [bs, N, T]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [bs, T, N, 1]
        
        # 如果节点数与模型训练时不同，需要传递节点索引
        if N != self.mtgnn.num_nodes:
            node_idx = torch.arange(N).to(self.args.device) % self.mtgnn.num_nodes
            output = self.mtgnn(x, idx=node_idx)
        else:
            output = self.mtgnn(x)
        
        # 输出形状为 [batch_size, out_dim, num_nodes, 1]
        # 重塑为 [batch_size * num_nodes, out_dim]
        output = output.squeeze(-1).permute(0, 2, 1)  # [bs, N, out_dim]
        output = output.reshape((-1, self.args.gcn["out_channel"]))
        return output

    


class TrafficEvent(nn.Module):
    def __init__(self, args):
        super(TrafficEvent, self).__init__()
        self.args = args
        
        self.extra_feature = args.extra_feature
        # self.basic_model = Basic_Model_org(args)
        self.basic_model = Basic_Model_org(args)
        
        self.memory_size = args.memory_size
        self.memory_dim = args.gcn["in_channel"]
        self.memory = nn.Parameter(torch.randn(self.memory_size, self.memory_dim))


        nn.init.xavier_uniform_(self.memory)

        if self.extra_feature:
            self.query_proj = nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"])
            self.extra_query_proj = nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"])
        
            self.classifier = nn.Sequential(
                nn.Linear(args.gcn["in_channel"] * (2+1), args.classifier["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(args.classifier["hidden_dim"], 2)
            )
        else:
            self.query_proj = nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"])
            self.classifier = nn.Sequential(
                nn.Linear(args.gcn["in_channel"] * 2, args.classifier["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(args.classifier["hidden_dim"], 2)
            )
        
        # create momentum models
        self.basic_model_m = Basic_Model_org(args)
        self.model_pairs = [[self.basic_model, self.basic_model_m]]
        
        self.copy_params()

    '''
    copied from albef 
    https://github.com/salesforce/ALBEF/blob/main/models/model_pretrain.py
    '''

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self, sim):
        if sim:
            if self.args.momentum_type == "linear":
                momentum = 0.9 + 0.05 * (sim - 0.5) * 10 
                momentum = torch.clamp(momentum, min=0.8, max=0.999) 
                # self.args.logger.info(f"momentum: {momentum}")
            elif self.args.momentum_type == "expo":
                base_momentum = 0.9
                momentum = 1 - (1 - base_momentum) * np.exp(-5 * sim.detach().cpu().numpy())
                momentum = np.clip(momentum, 0.8, 0.999)
            elif self.args.momentum_type == "constant":
                momentum = 0.995
            elif self.args.momentum_type == "wo":
                pass
            else:
                raise ValueError(f"momentum_type {self.args.momentum_type} not supported")

            if self.args.momentum_type != "wo":
                for model_pair in self.model_pairs:           
                    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                        param_m.data = param_m.data * momentum + param.data * (1. - momentum)
        else:
            # self.args.logger.info("No momentum update in EWC")
            pass
        
                
    
    def query_memory(self, x, adj):
        batch_size, N, feature_dim = x.shape
        
        if self.extra_feature:
            query_x = x  # [bs, N, in_channel]
            query_x_flat = query_x.reshape(-1, query_x.shape[-1])  # [bs*N, in_channel]

            query = self.query_proj(query_x_flat[..., :self.args.gcn["in_channel"]])  # [bs*N, in_channel*]
            query = query + self.extra_query_proj(query_x_flat[..., self.args.gcn["in_channel"]:])  # [bs*N, in_channel]
        
            query = query.reshape(batch_size, N, self.args.gcn["in_channel"])  # [bs, N, in_channel*2]
            query_avg = torch.mean(query, dim=1)  # [bs, in_channel*2]
        else:
            query_x = x[:, :, :self.args.gcn["in_channel"]]  # [bs, N, in_channel]
            query_x_flat = query_x.reshape(-1, query_x.shape[-1])  # [bs*N, in_channel]
            query = self.query_proj(query_x_flat)  # [bs*N, in_channel]
            query = query.reshape(batch_size, N, self.args.gcn["in_channel"])  # [bs, N, in_channel]
            query_avg = torch.mean(query, dim=1)  # [bs, in_channel]
        

        energy = torch.matmul(query_avg, self.memory.t())  # [bs, memory_size]
        attention = torch.softmax(energy, dim=-1)  # [bs, memory_size]
        memory_features = torch.matmul(attention, self.memory)  # [bs, in_channel]
        

        cos_similarity = F.cosine_similarity(query_avg, memory_features, dim=1).unsqueeze(1)  # [bs, 1]

        x_avg = torch.mean(x, dim=1)  # [bs, feature_dim]
        concat_features = torch.cat([x_avg, memory_features], dim=-1)  # [bs, in_channel*3]
        logits = self.classifier(concat_features)  # [bs, 2]
        # self.args.logger.info(f"cos_similarity: {cos_similarity[:5]}")

        node_memory_features = memory_features.unsqueeze(1).expand(-1, N, -1)  # [bs, N, in_channel]
        
        return cos_similarity, logits, node_memory_features
        
    def forward(self, data, adj):
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N

        if self.extra_feature:
            # 对于 extra_feature=True，输入特征维度是两倍
            x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]*2))  # [bs, N, feature*2]
        else:
            x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]*2))[:, :, :self.args.gcn["in_channel"]]  # [bs, N, feature]
        
        similarity, logits, node_memory_features = self.query_memory(x, adj)
        similarity = similarity.mean()


        from types import SimpleNamespace
        basic_data = SimpleNamespace()
        if self.args.expand:
            basic_data.x = torch.cat([data.x.reshape(-1, self.args.gcn["in_channel"]*2)[..., :self.args.gcn["in_channel"]], node_memory_features.reshape(-1,self.args.gcn["in_channel"])], dim=-1)
        else:
            basic_data.x = data.x.reshape(-1, self.args.gcn["in_channel"]*2)[:, :self.args.gcn["in_channel"]]


        # get momentum features
        with torch.no_grad():
            self._momentum_update(similarity)
            basic_features_m = self.basic_model_m(basic_data, adj)
        'this model do not need grads but the output need grads' 
        basic_features_m = basic_features_m.requires_grad_(True)  
        

        basic_features = self.basic_model(basic_data, adj)
        

        return basic_features, basic_features_m, similarity, logits
    
    def feature(self, data, adj):
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N

        if self.extra_feature:
            # 对于 extra_feature=True，输入特征维度是两倍
            x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]*2))  # [bs, N, feature*2]
        else:
            x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]*2))[:, :, :self.args.gcn["in_channel"]]  # [bs, N, feature]
        
        similarity, logits, node_memory_features = self.query_memory(x, adj)
        similarity = similarity.mean()


        from types import SimpleNamespace
        basic_data = SimpleNamespace()
        if self.args.expand:
            basic_data.x = torch.cat([data.x.reshape(-1, self.args.gcn["in_channel"]*2)[..., :self.args.gcn["in_channel"]], node_memory_features.reshape(-1,self.args.gcn["in_channel"])], dim=-1)
        else:
            basic_data.x = data.x.reshape(-1, self.args.gcn["in_channel"]*2)[:, :self.args.gcn["in_channel"]]


        basic_features = self.basic_model(basic_data, adj)

        return basic_features

    