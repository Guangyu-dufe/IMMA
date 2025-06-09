import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_conv import BatchGCNConv
from .GraphWaveNet import GWNET

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

class TrafficEvent(nn.Module):
    def __init__(self, args):
        super(TrafficEvent, self).__init__()
        self.args = args
        
        self.extra_feature = args.extra_feature
        # self.basic_model = Basic_Model_org(args)
        self.basic_model = GWNET_Model(args)
        
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

    