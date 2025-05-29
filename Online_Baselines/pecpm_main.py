import sys, argparse, random, torch
sys.path.append("src/")

import numpy as np
import os.path as osp
import networkx as nx

from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from utils.data_convert import generate_samples
from src.model.model import TrafficStream_Model, STKEC_Model, PECPM_Model
from dataer.PecpmDataset import TrafficDataset
from src.model import detect_pecpm
from src.model import replay

from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs, load_best_model
from src.trainer.pecpm_trainer import train, test_model


def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {"3":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "6":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "12":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "Avg":{" MAE":{}, "MAPE":{}, "RMSE":{}}}
    mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):  # 遍历从开始年份到结束年份的每一年
        # 加载图数据
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        
        # 根据data_process标志选择是否处理数据或直接加载数据
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=False) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+".npz"), allow_pickle=True)
        
        args.logger.info("[*] Year {} load from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        
        # 归一化邻接矩阵
        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        
        # 初始化node_list
        vars(args)["node_list"] = np.array([])
        
        # 如果是第一年且需要跳过第一年，模型已经训练过，不需要重新训练
        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            continue
        
        # 如果是增量策略且年份大于开始年份
        if year > args.begin_year and args.strategy == "incremental":
            model, _ = load_best_model(args)
            
            node_list = list()
            
            if args.increase:  # 获取新增节点
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))
            
            if args.detect:  # 获取受影响的节点
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
            
                # PECPM特有的节点检测逻辑
                evo_num = int(0.01*args.graph_size)
                replay_num = int(0.09*args.graph_size)  
                replay_nodes, evo_nodes = detect_pecpm.get_eveloved_nodes(args, replay_num, evo_num)
                node_list.extend(list(evo_nodes))
                node_list.extend(list(replay_nodes))
            
            node_list = list(set(node_list))
            if len(node_list) < int(0.1*args.graph_size):
                res = int(0.1 * args.graph_size) - len(node_list)
                res_node = [a for a in range(cur_node_size) if a not in node_list]
                if len(res_node) > 0:
                    expand_node_list = random.sample(res_node, min(res, len(res_node)))
                    node_list.extend(list(expand_node_list))
            
            # 获取节点列表的子图
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)  # 获取当前年份的边索引
            edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)  # 获取当前年份的图边列表
            
            graph_node_from_edge = set()  # 收集所有由边连接的节点
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            
            node_list = list(set(node_list) & graph_node_from_edge)  # 获取子图中的节点列表，即要修改的节点与已有变化节点的交集
            
            # 如果节点列表不为空
            if len(node_list) != 0:
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph  # 存储子图
                vars(args)["subgraph_edge_index"] = subgraph_edge_index  # 存储子图边索引
                vars(args)["mapping"] = mapping  # 存储节点映射
                args.logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format(len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            else:
                args.logger.info("number of increase nodes:{}, total nodes this year {}".format(len(node_list), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)

        # 当没有需要增量训练的节点时，跳过这一年
        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)  # 加载最佳模型
            mkdirs(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year), loss+".pkl"))
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            args.logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        
        if args.train:  # 如果需要训练
            train(inputs, args)
        else:
            if args.auto_test:  # 如果需要自动测试
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
                test_model(model, args, test_loader, pin_memory=True)
    
    # 打印每年不同步长的指标
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year+1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.2f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            if len(info_list) > 0:
                args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(info_list)))

    # 打印总训练时间、每轮平均训练时间和训练轮数
    total_time = 0
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(year, args.result[year]["total_time"], args.result[year]["average_time"], args.result[year]['epoch_num'])
            total_time += args.result[year]["total_time"]
            args.logger.info(info)
    args.logger.info("total time: {:.4f}".format(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/SD/pecpm.json")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--logname", type = str, default = "pecpm")
    parser.add_argument("--method", type = str, default = "PECPM")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "log/SD/pecpm-42/2017/31.5668.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["methods"] = {'TrafficStream': TrafficStream_Model, 'STKEC': STKEC_Model, 'PECPM': PECPM_Model}
    
    init(args)
    seed_anything(args.seed)
    init_log(args)
    
    main(args) 