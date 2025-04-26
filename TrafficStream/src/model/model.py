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
        
        # 不考虑 extra_feature，假设输入的 data.x 已经是正确的维度 [bs*N, in_channel]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
            
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        # 直接使用原始输入进行残差连接
        x = x + data.x
            
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

 
    def feature(self, data, adj):
        N = adj.shape[0]
        
        # 不考虑 extra_feature，假设输入的 data.x 已经是正确的维度 [bs*N, in_channel]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
            
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        # 直接使用原始输入进行残差连接
        x = x + data.x
            
        return x


class TrafficEvent(nn.Module):
    def __init__(self, args):
        super(TrafficEvent, self).__init__()
        self.args = args
        
        # 设置 extra_feature 属性
        self.extra_feature = True
        args.extra_feature = self.extra_feature  # 确保传递给 Basic_Model
        
        self.basic_model = Basic_Model(args)
        # self.event_model = Basic_Model(args)
        
        self.memory_size = args.memory_size if hasattr(args, 'memory_size') else 32
        self.memory_dim = args.gcn["in_channel"]
        self.memory = nn.Parameter(torch.randn(self.memory_size, self.memory_dim))
        
        # 不管是否使用扩展特征，query_proj 都只接收原始维度的输入
        self.query_proj = nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"])
        
        # 使用类属性而不是局部变量
        if self.extra_feature:
            self.classifier = nn.Sequential(
                nn.Linear(args.gcn["in_channel"] * (2+1), 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(args.gcn["in_channel"] * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )

    def query_memory(self, x, adj):
        batch_size, N, feature_dim = x.shape
        
        # 如果使用扩展特征，只取前一半用于内存查询
        if self.extra_feature:
            # 只使用前一半特征进行查询
            query_x = x[:, :, :self.args.gcn["in_channel"]]  # [bs, N, in_channel]
        else:
            query_x = x  # [bs, N, in_channel]
            
        query_x_flat = query_x.reshape(-1, query_x.shape[-1])  # [bs*N, in_channel]
        query = self.query_proj(query_x_flat)  # [bs*N, in_channel]
        query = query.reshape(batch_size, N, self.args.gcn["in_channel"])  # [bs, N, in_channel]
        
        query_avg = torch.mean(query, dim=1)  # [bs, in_channel]
        
        energy = torch.matmul(query_avg, self.memory.t())  # [bs, memory_size]
        attention = torch.softmax(energy, dim=-1)  # [bs, memory_size]
        
        memory_features = torch.matmul(attention, self.memory)  # [bs, in_channel]
        
        # 对于分类，仍使用完整的特征
        x_avg = torch.mean(x, dim=1)  # [bs, feature_dim]
        
        # 根据 extra_feature 构建不同的特征组合
        if self.extra_feature:
            # 使用三种特征：原始特征的前一半 + 内存特征 + 原始特征的前一半
            x_front_half = x_avg[:, :self.args.gcn["in_channel"]]  # [bs, in_channel]
            concat_features = torch.cat([x_front_half, memory_features, x_front_half], dim=-1)  # [bs, in_channel*3]
        else:
            concat_features = torch.cat([x_avg, memory_features], dim=-1)  # [bs, in_channel*2]
        
        logits = self.classifier(concat_features)  # [bs, 2]
        
        return memory_features, logits
        
    def forward(self, data, adj):
        N = adj.shape[0]
        batch_size = data.x.shape[0] // N
        
        # 重塑输入数据，适应不同的 extra_feature 设置
        if self.extra_feature:
            # 对于 extra_feature=True，输入特征维度是两倍
            x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]*2))  # [bs, N, feature*2]
        else:
            # 对于 extra_feature=False，使用原始特征维度
            x = data.x.reshape((batch_size, N, self.args.gcn["in_channel"]))  # [bs, N, feature]
        
        # 内存查询模块 - 仅使用前一半特征进行查询，但在分类时使用特征组合
        memory_features, logits = self.query_memory(x, adj)
        
        # 为 basic_model 准备数据，只保留前一半特征
        if self.extra_feature:
            # 创建一个新的数据容器，只包含前一半特征
            from types import SimpleNamespace
            basic_data = SimpleNamespace()
            
            # 只提取前一半特征维度 [bs*N, in_channel]
            reshaped_x = data.x.reshape(-1, self.args.gcn["in_channel"]*2)
            basic_data.x = reshaped_x[:, :self.args.gcn["in_channel"]]
        else:
            # 不需要特殊处理，直接使用原始数据
            basic_data = data
    
        # 调用 basic_model 的 feature 方法处理前一半特征
        basic_features = self.basic_model.feature(basic_data, adj)
        
        return basic_features, memory_features, logits