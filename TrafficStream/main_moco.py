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
pin_memory = True
n_work = 16 

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
    model = TrafficEvent(args)
    model.load_state_dict(state_dict)
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


def train(inputs, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    if args.loss == "mse": lossfunc = func.mse_loss
    elif args.loss == "huber": lossfunc = func.smooth_l1_loss

    # Dataset Definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(TrafficDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work) 
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
        val_loader = DataLoader(TrafficDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        if args.ewc:
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)
            ewc_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
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


    num_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    args.logger.info(f"[*] Classifier parameter count: {num_params}")

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
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        similaritys = []
        # Train Model
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(device, non_blocking=pin_memory)
            
            optimizer.zero_grad()

            basic_features, basic_features_m, similarity, logits = model(data, args.sub_adj)
            similaritys.append(float(similarity.detach().cpu().numpy()))
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]

            # compute loss
            predictions = torch.argmax(logits, dim=-1)  # [bs]
            loss_basic = lossfunc(data.y, basic_features, reduction="none")
            # loss_event = lossfunc(data.y, event_features, reduction="none")
            loss_basic = loss_basic.reshape(len(predictions), -1 ,basic_features.shape[1]).mean(dim=1).mean(dim=1)
            # loss_event = loss_event.reshape(len(logits), -1 ,event_features.shape[1]).mean(dim=1).mean(dim=1)

            q3 = torch.quantile(loss_basic, 0.75)
            loss_mask = torch.zeros_like(loss_basic, dtype=torch.long)
            loss_mask[loss_basic > q3] = 1
            loss_classification = F.cross_entropy(logits, loss_mask)


            loss_basic_m = lossfunc(basic_features, basic_features_m, reduction="mean")
            
            loss = loss_classification + loss_basic.mean() + loss_basic_m*0.3

            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()
            training_loss += float(loss)
            classification_loss += float(loss_classification)
            basic_loss += float(loss_basic[predictions==0].mean())
            event_loss += float(loss_basic[predictions==1].mean())
            loss.backward()


            optimizer.step()
            
            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn
        basic_loss = basic_loss/cn
        event_loss = event_loss/cn
        classification_loss = classification_loss/cn

        # Validate Model
        validation_loss = 0.0
        validation_basic_loss = 0.0
        validation_event_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device,non_blocking=pin_memory)
                basic_features, basic_features_m, memory_features, logits = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                predictions = torch.argmax(logits, dim=-1)  # [bs]
                loss_basic = lossfunc(data.y, basic_features, reduction="none")
                loss_basic = loss_basic.reshape(len(predictions), -1 ,basic_features.shape[1]).mean(dim=1).mean(dim=1)
                loss = masked_mae_np(data.y.cpu().data.numpy(), basic_features.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                validation_basic_loss += float(loss_basic[predictions==0].mean())
                validation_event_loss += float(loss_basic[predictions==1].mean())
                cn += 1
        
        validation_loss = float(validation_loss/cn)
        validation_basic_loss = validation_basic_loss/cn
        validation_event_loss = validation_event_loss/cn
        # For any grouped data, the overall average must be less than or equal to the sum of the average values ​​of each group.
        # args.logger.info("[*]Train--basic_loss:{:.4f}, event_loss:{:.4f}, classification_loss:{:.4f}".format(basic_loss, event_loss, classification_loss))
        # args.logger.info("[*]Validation--basic_loss:{:.4f}, event_loss:{:.4f}".format(validation_basic_loss, validation_event_loss))

        # args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}, similarity:{similaritys[:5]}")
        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}, moco loss:{loss_basic_m.mean():.4f}")

        # Early Stop
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

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
    pred_ = []
    truth_ = []
    test_loss = 0.0
    test_basic_loss = 0.0
    test_event_loss = 0.0
    test_classification_loss = 0.0
    with torch.no_grad():
        cn = 0
        pred_basic = []
        truth_basic = []
        pred_event = []
        truth_event = []
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            basic_features, basic_features_m, memory_features, logits = model(data, args.sub_adj)
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]
            predictions = torch.argmax(logits, dim=-1)  # [bs]
            loss_basic = func.mse_loss(data.y, basic_features, reduction="none")
            loss_basic = loss_basic.reshape(len(predictions), -1, basic_features.shape[1]).mean(dim=1).mean(dim=1)
    
            test_loss += float(loss_basic.mean())
            test_basic_loss += float(loss_basic[predictions==0].mean())
            test_event_loss += float(loss_basic[predictions==1].mean())
            
            # Save samples with prediction=0 and prediction=1 separately
            basic_mask = predictions == 0
            event_mask = predictions == 1
            
            # Get number of nodes per batch
            nodes_per_batch = int(basic_features.shape[0] // len(predictions))
            
            # Reshape features and labels for batch grouping
            features_reshaped = basic_features.reshape(len(predictions), nodes_per_batch, -1)
            labels_reshaped = data.y.reshape(len(predictions), nodes_per_batch, -1)
            
            # Save predictions and ground truth for basic traffic and event traffic separately
            if torch.any(basic_mask):
                pred_basic.append(features_reshaped[basic_mask].cpu().data.numpy())
                truth_basic.append(labels_reshaped[basic_mask].cpu().data.numpy())
            
            if torch.any(event_mask):
                pred_event.append(features_reshaped[event_mask].cpu().data.numpy())
                truth_event.append(labels_reshaped[event_mask].cpu().data.numpy())
            
            # Save all samples for overall evaluation
            pred_.append(basic_features.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            
            cn += 1
        
        test_loss = test_loss/cn
        test_basic_loss = test_basic_loss/cn
        test_event_loss = test_event_loss/cn
        
        args.logger.info("[*]Test--basic_loss:{:.4f}, event_loss:{:.4f}, test_loss:{:.4f}".format(
            test_basic_loss, test_event_loss, test_loss))
        
        # Process all samples for evaluation
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        nodes_per_batch = int(basic_features.shape[0] // len(predictions))
        pred_ = pred_.reshape(-1, nodes_per_batch, pred_.shape[1])
        truth_ = truth_.reshape(-1, nodes_per_batch, truth_.shape[1])
        
        # Evaluate all samples
        args.logger.info("[*] Evaluating all samples:")
        mae_all = metric(truth_, pred_, args)
        
        # Evaluate basic traffic samples
        if pred_basic:
            pred_basic = np.concatenate(pred_basic, 0)
            truth_basic = np.concatenate(truth_basic, 0)
            args.logger.info("[*] Evaluating basic traffic samples (prediction=0):")
            mae_basic = metric(truth_basic, pred_basic, args)
        
        # Evaluate event traffic samples
        if pred_event:
            pred_event = np.concatenate(pred_event, 0)
            truth_event = np.concatenate(truth_event, 0)
            args.logger.info("[*] Evaluating event traffic samples (prediction=1):")
            mae_event = metric(truth_event, pred_event, args)
        
        return test_loss


def metric(ground_truth, prediction, args):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        # Load Data 
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), 
                                  np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"],
                                  np.load(osp.join(args.raw_data_path, str(year)+".npz"))["event_type_code"],
                                  graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)

        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year)))) 

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        if year == args.begin_year and args.load_first_year:
            # Skip the first year, model has been trained and retrain is not needed
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
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
                vars(args)["topk"] = int(0.01*args.graph_size) 
                influence_node_list = detect.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                node_list.extend(list(influence_node_list))

            # Obtain sample nodes
            if args.replay:
                vars(args)["replay_num_samples"] = int(0.09*args.graph_size) #int(0.2*args.graph_size)- len(node_list)
                args.logger.info("[*] replay node number {}".format(args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(args, inputs, model)
                node_list.extend(list(replay_node_list))
            
            node_list = list(set(node_list))
            if len(node_list) > int(0.1*args.graph_size):
                node_list = random.sample(node_list, int(0.1*args.graph_size))
            
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

        
        # Skip the year when no nodes needed to be trained incrementally
        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
            test_model(model, args, test_loader, pin_memory=True)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        

        if args.train:
            train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
                test_model(model, args, test_loader, pin_memory=True)


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
    parser.add_argument("--conf", type = str, default = "conf/trafficStream_SD.json")
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 5)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "/home/bd2/ANATS/TrafficStream/res/SD/trafficStream2025-04-17-12:25:53.499043/2017/26.012.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    seed_set(13)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)