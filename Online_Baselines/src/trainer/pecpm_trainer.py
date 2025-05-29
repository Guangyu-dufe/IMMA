import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch

from ..model.ewc import EWC
from torch_geometric.loader import DataLoader
from ..dataer.PecpmDataset import TrafficDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model


def train(inputs, args):
    path = osp.join(args.path, str(args.year))  # 定义当前年份模型保存路径
    mkdirs(path)
    
    # 设置损失函数
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss
    
    cluster_lossfunc = func.mse_loss  # PECPM使用MSE作为attention损失
    
    # 数据集定义
    N = inputs['train_x'].shape[-1]
    
    # 加载attention数据
    pathatt = 'data/SD/attetion/' + str(args.year) + '_attention.npy'
    attention = np.load(pathatt)
    C = attention.shape[-1]
    attention = attention.reshape(-1, N, C)

    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(TrafficDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()],\
            att=attention[:, args.subgraph.numpy(),:],edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(TrafficDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
             att=attention[:, args.subgraph.numpy(),:],edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32) 
        test_loader = DataLoader(TrafficDataset("", "", x=inputs["test_x"][:, :, args.subgraph.numpy()], y=inputs["test_y"][:, :, args.subgraph.numpy()], \
             att=attention[:, args.subgraph.numpy(),:],edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        train_loader = DataLoader(TrafficDataset(inputs, "train", att=attention), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(TrafficDataset(inputs, "val", att=attention), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        test_loader = DataLoader(TrafficDataset(inputs, "test", att=attention), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        vars(args)["sub_adj"] = vars(args)["adj"]

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # 模型定义
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)  # 如果不是第一年，加载最优模型
        if args.ewc:  # 如果使用ewc策略，使用ewc模型
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # 记录EWC相关参数
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # 初始化EWC模型
            ewc_loader = DataLoader(TrafficDataset(inputs, "train", att=attention), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            model.register_ewc_params_for_pecpm(ewc_loader, lossfunc, args.device)  # 注册EWC参数
        else:
            model = gnn_model  # 否则使用加载的最佳模型
    else:
        gnn_model = args.methods[args.method](args).to(args.device)  # 如果是第一年，使用基础模型
        model = gnn_model
    
    # 模型优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    
    for epoch in range(args.epoch):
        
        start_time = datetime.now()
        
        # 训练模型
        cn = 0
        training_loss = 0.0
        loss_cluster = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            
            pred, attention_pred = model(data, args.sub_adj)
            
            # PECPM的attention损失计算
            attention_label = data.att.to(args.device)
            
            loss_cluster = cluster_lossfunc(attention_pred, attention_label)
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch用于将一批稀疏邻接矩阵转换为一批密集邻接矩阵
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]  # 根据映射进行切片，获得变化节点的预测和真实值
                data.y = data.y[:, args.mapping, :]
            
            loss = lossfunc(data.y, pred, reduction="mean") + loss_cluster * args.beita  # 计算损失
            
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()  # 计算并添加ewc损失
            
            training_loss += float(loss)
            cn += 1
            
            loss.backward()
            optimizer.step()
        
        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn 
        
        # 验证模型
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred, attention_pred = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                    
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        # 早停策略
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            if args.ewc:
                torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            else:
                torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")  # 选择验证损失最低的模型作为最优模型
    best_model = args.methods[args.method](args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # 测试模型
    test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            # 在增量学习模式下使用sub_adj，否则使用完整的adj
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, attention = model(data, args.sub_adj)
            else:
                pred, attention = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            # 在增量学习模式下应用映射处理
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args) 