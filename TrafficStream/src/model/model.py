import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gcn_conv import BatchGCNConv


class Basic_Model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.args = args

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


class TrafficEvent(nn.Module):
    def __init__(self, args):
        super(TrafficEvent, self).__init__()
        self.args = args
        self.basic_model = Basic_Model(args)
        # self.event_model = Basic_Model(args)
        
        self.memory_size = args.memory_size if hasattr(args, 'memory_size') else 32
        self.memory_dim = args.gcn["in_channel"]
        self.memory = nn.Parameter(torch.randn(self.memory_size, self.memory_dim))
        
        self.query_proj = nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"])
        
        self.classifier = nn.Sequential(
            nn.Linear(args.gcn["in_channel"] * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def query_memory(self, x, adj):
        batch_size, N, feature_dim = x.shape
        
        x_flat = x.reshape(-1, feature_dim)  # [bs*N, feature]
        query = self.query_proj(x_flat)  # [bs*N, feature]
        query = query.reshape(batch_size, N, feature_dim)  # [bs, N, feature]
        
        query_avg = torch.mean(query, dim=1)  # [bs, feature]
        
        energy = torch.matmul(query_avg, self.memory.t())  # [bs, memory_size]
        attention = torch.softmax(energy, dim=-1)  # [bs, memory_size]
        
        memory_features = torch.matmul(attention, self.memory)  # [bs, feature]
        
        x_avg = torch.mean(x, dim=1)  # [bs, feature]
        concat_features = torch.cat([x_avg, memory_features], dim=-1)  # [bs, feature*2]
        
        logits = self.classifier(concat_features)  # [bs, 2]
        
        return memory_features, logits
        
    def forward(self, data, adj):
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        
        x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]))  # [bs, N, feature]
        
        memory_features, logits = self.query_memory(x, adj)
    
        basic_features = self.basic_model.feature(data, adj)
        # event_features = self.event_model.feature(data, adj)
        
        return basic_features, memory_features, logits