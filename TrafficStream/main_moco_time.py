import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb
import gc  
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from utils.data_convert import generate_samples
from src.model.model import TrafficEvent
from src.model.ewc import EWC
from src.trafficDataset import TrafficDataset
from src.model import detect
from src.model import replay
import torch.nn.functional as F

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = False
n_work = 4

def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year+1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))): 
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+args.time, str(args.year-1), loss[0]+".pkl")
        
    args.logger.info("Secure SignIn[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    if 'tcn2.weight' in state_dict:
        del state_dict['tcn2.weight']
        del state_dict['tcn2.bias']
    model = TrafficEvent(args).to(args.device)

    # very not safe, and it's only for 2017 year
    x = torch.randn(128*args.graph_size, 24).to(args.device)
    adj = torch.randn(args.graph_size, args.graph_size).to(args.device)

    data = Data(x=x)
    model(data, adj)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    return model, loss[0]

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def get_memory_usage():
    """获取当前内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    else:
        import psutil
        return psutil.virtual_memory().used / 1024**3  # GB

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train(inputs, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    if args.loss == "mse": lossfunc = func.mse_loss
    elif args.loss == "huber": lossfunc = func.smooth_l1_loss

    # 根据数据大小动态调整batch size
    # data_size = inputs["train_x"].shape[0] if hasattr(inputs["train_x"], 'shape') else len(inputs["train_x"])
    # if data_size > 10000:  # 如果数据很大，减小batch size
    #     dynamic_batch_size = max(16, args.batch_size // 2)
    #     args.logger.info(f"[*] Large dataset detected ({data_size} samples), reducing batch size from {args.batch_size} to {dynamic_batch_size}")
    # else:
    dynamic_batch_size = args.batch_size

    # Dataset Definition - 使用动态batch size
    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(TrafficDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=dynamic_batch_size, shuffle=True, pin_memory=False, num_workers=4)
        val_loader = DataLoader(TrafficDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=dynamic_batch_size, shuffle=False, pin_memory=False, num_workers=4) 
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=dynamic_batch_size, shuffle=True, pin_memory=False, num_workers=4)
        val_loader = DataLoader(TrafficDataset(inputs, "val"), batch_size=dynamic_batch_size, shuffle=False, pin_memory=False, num_workers=4)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=dynamic_batch_size, shuffle=False, pin_memory=False, num_workers=4)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")
    args.logger.info(f"[*] Initial memory usage: {get_memory_usage():.2f} GB")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        if args.ewc:
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)
            ewc_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=4)
            model.register_ewc_params(ewc_loader, lossfunc, device)
        else:
            model = gnn_model
    else:
        gnn_model = TrafficEvent(args).to(args.device)
        model = gnn_model

    # Model Optimizer
    num_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
    args.logger.info(f"[*] Model parameter count: {num_params}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # num_params = sum(p.numel() for p in model.memory.parameters() if p.requires_grad)
    # args.logger.info(f"[*] Memory parameter count: {num_params}")


    # num_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    # args.logger.info(f"[*] Classifier parameter count: {num_params}")

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    global_train_steps = len(train_loader) // args.batch_size +1

    iters = len(train_loader)
    lowest_validation_loss = 1e7
    classification_loss = 0
    basic_loss = 0
    event_loss = 0
    counter = 0
    patience = 5
    model.train()
    use_time = []
    
    # 统计所有epoch的forward和backward时间
    total_forward_time = 0.0
    total_backward_time = 0.0
    completed_epochs = 0
    
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        similaritys = []
        moco_loss_sum = 0.0
        # Train Model
        cn = 0
        
        # 记录forward和backward时间
        forward_time = 0.0
        backward_time = 0.0
        
        # 每10个epoch清理一次内存
        if epoch % 10 == 0:
            clear_memory()
            args.logger.info(f"[*] Epoch {epoch}, memory usage: {get_memory_usage():.2f} GB")
        
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(device, non_blocking=False)
            
            optimizer.zero_grad()

            # 开始计时forward pass
            forward_start = datetime.now()
            basic_features, basic_features_m, similarity, logits = model(data, args.sub_adj)
            forward_end = datetime.now()
            forward_time += (forward_end - forward_start).total_seconds()
            
            similaritys.append(float(similarity.detach().cpu().numpy()))
            
            # only need for number of node is not same
            if args.strategy == "incremental" and args.year > args.begin_year:
                basic_features_m, _ = to_dense_batch(basic_features_m, batch=data.batch)
                basic_features, _ = to_dense_batch(basic_features, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                basic_features = basic_features[:, args.mapping, :]
                basic_features_m = basic_features_m[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]

            # compute loss (不计时这部分)
            predictions = torch.argmax(logits, dim=-1)  # [bs]
            loss_basic = lossfunc(data.y, basic_features, reduction="none")
            # loss_event = lossfunc(data.y, event_features, reduction="none")
            loss_basic = loss_basic.reshape(len(predictions), -1 ,basic_features.shape[1]).mean(dim=1).mean(dim=1)
            # loss_event = loss_event.reshape(len(logits), -1 ,event_features.shape[1]).mean(dim=1).mean(dim=1)

            q3 = torch.quantile(loss_basic, 0.5)
            loss_mask = torch.zeros_like(loss_basic, dtype=torch.long)
            loss_mask[loss_basic > q3] = 1
            loss_classification = F.cross_entropy(logits, loss_mask)

            loss_basic_m = lossfunc(basic_features, basic_features_m, reduction="mean")

            loss = loss_basic.mean() + loss_basic_m*0.2

            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()
            training_loss += float(loss)
            classification_loss += float(loss_classification)
            basic_loss += float(loss_basic[predictions==0].mean())
            event_loss += float(loss_basic[predictions==1].mean())
            
            # 保存moco loss用于记录
            moco_loss_value = float(loss_basic_m.mean())
            moco_loss_sum += moco_loss_value
            
            # 开始计时backward pass
            backward_start = datetime.now()
            loss.backward()
            optimizer.step()
            backward_end = datetime.now()
            backward_time += (backward_end - backward_start).total_seconds()
            
            # 定期清理中间变量
            del loss_basic, loss_classification, loss_basic_m, loss, predictions
            
            # 每50个batch清理一次GPU缓存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            cn += 1

        # 累积每个epoch的forward和backward时间
        total_forward_time += forward_time
        total_backward_time += backward_time
        completed_epochs += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn
        basic_loss = basic_loss/cn
        event_loss = event_loss/cn
        classification_loss = classification_loss/cn
        moco_loss_avg = moco_loss_sum/cn

        # Validate Model
        validation_loss = 0.0
        validation_basic_loss = 0.0
        validation_event_loss = 0.0
        cn = 0
        model.eval()  # 确保模型在验证模式
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device,non_blocking=False)  # 关闭non_blocking
                basic_features, basic_features_m, memory_features, logits = model(data, args.sub_adj)

                # only need for number of node is not same
                if args.strategy == "incremental" and args.year > args.begin_year:
                    basic_features_m, _ = to_dense_batch(basic_features_m, batch=data.batch)
                    basic_features, _ = to_dense_batch(basic_features, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    basic_features = basic_features[:, args.mapping, :]
                    basic_features_m = basic_features_m[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]

                predictions = torch.argmax(logits, dim=-1)  # [bs]
                loss_basic = lossfunc(data.y, basic_features, reduction="none")
                loss_basic = loss_basic.reshape(len(predictions), -1 ,basic_features.shape[1]).mean(dim=1).mean(dim=1)
                loss = masked_mae_np(data.y.cpu().data.numpy(), basic_features.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                if len(loss_basic[predictions==0]) > 0:
                    validation_basic_loss += float(loss_basic[predictions==0].mean())
                if len(loss_basic[predictions==1]) > 0:
                    validation_event_loss += float(loss_basic[predictions==1].mean())
                
                # 显式释放内存
                del basic_features, basic_features_m, data, loss_basic
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                cn += 1
        
        model.train()  # 恢复训练模式
        validation_loss = float(validation_loss/cn)
        validation_basic_loss = validation_basic_loss/cn
        validation_event_loss = validation_event_loss/cn
        # For any grouped data, the overall average must be less than or equal to the sum of the average values ​​of each group.
        # args.logger.info("[*]Train--basic_loss:{:.4f}, event_loss:{:.4f}, classification_loss:{:.4f}".format(basic_loss, event_loss, classification_loss))
        # args.logger.info("[*]Validation--basic_loss:{:.4f}, event_loss:{:.4f}".format(validation_basic_loss, validation_event_loss))

        # args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}, similarity:{similaritys[:5]}")
        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}, moco loss:{moco_loss_avg:.4f}, forward time:{forward_time:.4f}s, backward time:{backward_time:.4f}s")

        # Early Stop
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            # 保存后清理缓存
            clear_memory()
        else:
            counter += 1
            if counter > 2:
                break

    # 计算平均forward和backward时间
    avg_forward_time = total_forward_time / completed_epochs if completed_epochs > 0 else 0.0
    avg_backward_time = total_backward_time / completed_epochs if completed_epochs > 0 else 0.0

    # 训练结束后清理内存
    clear_memory()
    args.logger.info(f"[*] Training completed, memory usage: {get_memory_usage():.2f} GB")
    args.logger.info(f"[*] Year {args.year} Training Summary - Completed epochs: {completed_epochs}")
    args.logger.info(f"[*] Average forward time per epoch: {avg_forward_time:.4f}s")
    args.logger.info(f"[*] Average backward time per epoch: {avg_backward_time:.4f}s")
    args.logger.info(f"[*] Total forward time: {total_forward_time:.4f}s, Total backward time: {total_backward_time:.4f}s")

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = TrafficEvent(args)

    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # Test Model
    test_model(best_model, args, test_loader, pin_memory)
    result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    # 移除全局列表，改用在线计算
    test_loss = 0.0
    test_basic_loss = 0.0
    test_event_loss = 0.0
    
    # 用于累积度量计算的变量
    total_samples = 0
    mae_sum = {3: 0.0, 6: 0.0, 12: 0.0}
    rmse_sum = {3: 0.0, 6: 0.0, 12: 0.0}
    mape_sum = {3: 0.0, 6: 0.0, 12: 0.0}
    
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            basic_features, basic_features_m, similarity, logits = model(data, args.adj)

            # only need for number of node is not same
            if args.strategy == "incremental" and args.year > args.begin_year:
                basic_features_m, _ = to_dense_batch(basic_features_m, batch=data.batch)
                basic_features, _ = to_dense_batch(basic_features, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                basic_features = basic_features[:, args.mapping, :]
                basic_features_m = basic_features_m[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]
            
            predictions = torch.argmax(logits, dim=-1)  # [bs]
            loss_basic = func.mse_loss(data.y, basic_features, reduction="none")
            loss_basic = loss_basic.reshape(len(predictions), -1, basic_features.shape[1]).mean(dim=1).mean(dim=1)
    
            test_loss += float(loss_basic.mean())

            if len(loss_basic[predictions==0]) > 0:
                test_basic_loss += float(loss_basic[predictions==0].mean())
            if len(loss_basic[predictions==1]) > 0:
                test_event_loss += float(loss_basic[predictions==1].mean())
            
            # 在线计算度量而不是累积数据
            batch_size = basic_features.shape[0]
            nodes_per_batch = int(batch_size // len(predictions))
            
            # 重塑数据进行度量计算
            pred_reshaped = basic_features.reshape(len(predictions), nodes_per_batch, -1)
            truth_reshaped = data.y.reshape(len(predictions), nodes_per_batch, -1)
            
            # 转换为numpy进行度量计算
            pred_np = pred_reshaped.cpu().data.numpy()
            truth_np = truth_reshaped.cpu().data.numpy()
            
            # 在线累积度量值
            batch_samples = pred_np.shape[0]
            total_samples += batch_samples
            
            for i in [3, 6, 12]:
                if pred_np.shape[2] >= i:
                    mae = masked_mae_np(truth_np[:, :, :i], pred_np[:, :, :i], 0)
                    rmse = masked_mse_np(truth_np[:, :, :i], pred_np[:, :, :i], 0) ** 0.5
                    mape = masked_mape_np(truth_np[:, :, :i], pred_np[:, :, :i], 0)
                    
                    mae_sum[i] += mae * batch_samples
                    rmse_sum[i] += rmse * batch_samples
                    mape_sum[i] += mape * batch_samples
            
            # 显式删除大的tensor以释放内存
            del basic_features, basic_features_m, data, pred_reshaped, truth_reshaped, pred_np, truth_np
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            cn += 1
        
        test_loss = test_loss/cn
        test_basic_loss = test_basic_loss/cn
        test_event_loss = test_event_loss/cn

        args.logger.info("[*]Test--basic_loss:{:.4f}, event_loss:{:.4f}, test_loss:{:.4f}".format(
            test_basic_loss, test_event_loss, test_loss))
        
        # 计算平均度量值
        args.logger.info("[*] Evaluating all samples:")
        for i in [3, 6, 12]:
            if total_samples > 0:
                mae_avg = mae_sum[i] / total_samples
                rmse_avg = rmse_sum[i] / total_samples  
                mape_avg = mape_sum[i] / total_samples
                
                args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae_avg,rmse_avg,mape_avg))
                
                # 存储到全局结果
                result[i]["mae"][args.year] = mae_avg
                result[i]["mape"][args.year] = mape_avg
                result[i]["rmse"][args.year] = rmse_avg
        
        return test_loss


def metric(flag, ground_truth, prediction, args):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        if flag:
            result[i]["mae"][args.year] = mae
            result[i]["mape"][args.year] = mape
            result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        # 显式垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load Data 
        args.logger.info(f"[*] Loading data for year {year}, current memory usage: {get_memory_usage():.2f} GB")
        
        # 加载图数据
        adj_data = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
        graph = nx.from_numpy_array(adj_data)
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        
        # 优化数据生成/加载过程
        if args.data_process:
            # 分步加载大数据文件
            raw_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))
            data_array = raw_data["x"]
            event_array = raw_data["event_type_code"]
            inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), 
                                      data_array, event_array, graph, val_test_mix=True)
            # 立即删除原始数据以释放内存
            del raw_data, data_array, event_array
        else:
            inputs = np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)

        args.logger.info("[*] Year {} load from {}_30day.npz, memory usage: {:.2f} GB".format(args.year, osp.join(args.save_data_path, str(year)), get_memory_usage())) 

        # 处理邻接矩阵
        adj = adj_data / (np.sum(adj_data, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        del adj_data  # 删除原始邻接矩阵数据

        if year == args.begin_year and args.load_first_year:
            # Skip the first year, model has been trained and retrain is not needed
            adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
            adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
            vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=max(16, args.batch_size//2), shuffle=False, pin_memory=False, num_workers=4)

            # only for no debug
            test_model(model, args, test_loader, pin_memory=False)
            
            # 释放内存
            del model, inputs, adj, graph, test_loader
            clear_memory()
            continue

        
        if year > args.begin_year and args.strategy == "incremental":
            # Load the best model
            model, _ = load_best_model(args)
            
            node_list = list()
            # Obtain increase nodes
            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            # Obtain influence nodes
            if args.detect:
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
                # 20% of current graph size will be sampled
                if args.logname == "gba":
                    vars(args)["topk"] = 10
                else:
                    vars(args)["topk"] = int(0.01*args.graph_size) 
                influence_node_list = detect.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                node_list.extend(list(influence_node_list))

            # Obtain sample nodes
            if args.replay:
                # vars(args)["replay_num_samples"] = int(0.09*args.graph_size) #int(0.2*args.graph_size)- len(node_list)
                args.logger.info("[*] replay node number {}".format(args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(args, inputs, model)
                node_list.extend(list(replay_node_list))
            
            node_list = list(set(node_list))
            # if len(node_list) > int(0.1*args.graph_size):
            #     node_list = random.sample(node_list, int(0.1*args.graph_size))
            
           
            if args.logname == "gba":
                if len(node_list) < int(0.1*args.graph_size):
                    res=int(0.1 * args.graph_size)-len(node_list)
                    res_node = [a for a in range(cur_node_size) if a not in node_list]
                    expand_node_list = random.sample(res_node, res)
                    node_list.extend(list(expand_node_list))
                
            # Obtain subgraph of node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
            

            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)

        
        # # Skip the year when no nodes needed to be trained incrementally
        # if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
        #     model, loss = load_best_model(args)
        #     ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
        #     torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
        #     test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        #     test_model(model, args, test_loader, pin_memory=True)
        #     logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
        #     continue
        

        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=max(16, args.batch_size//2), shuffle=False, pin_memory=False, num_workers=4)
                test_model(model, args, test_loader, pin_memory=False)
                del test_loader
        
        # 每年结束后清理内存
        del inputs
        if 'model' in locals():
            del model
        if 'adj' in locals():
            del adj
        if 'graph' in locals():
            del graph
        clear_memory()
        args.logger.info(f"[*] Year {year} completed, memory usage: {get_memory_usage():.2f} GB")

    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            for year in range(args.begin_year, args.end_year+1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            info+="{:.2f}\t".format(result[i][j][year])
            logger.info("{}\t{}\t".format(i,j) + info)

    for year in range(args.begin_year, args.end_year+1):
        if year in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
            logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/gba.json")
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 7)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "/home/bd2/ANATS/TrafficStream/res/CA/ca2025-06-08-11:35:41.262744/2019/24.8698.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    seed_set(42)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)