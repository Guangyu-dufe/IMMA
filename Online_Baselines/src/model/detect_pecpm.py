import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch 
from dataer.PecpmDataset import continue_learning_Dataset
from torch_geometric.data import Data, Batch, DataLoader
import torch
from scipy.spatial import distance
from scipy.stats import wasserstein_distance as WD
import os.path as osp
import random


def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=3)
    # feature shape [T', feature_dim, N]
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)
        node_size = feature.size()[1]
        # print("before permute:", feature.size())
        feature = feature.permute(1,0,2)

        # [N, T', feature_dim]
        return feature.cpu().detach().numpy()


def sort_with_index(lst):
    indexed_list = [(value, index) for index, value in enumerate(lst)]
    sorted_list = sorted(indexed_list)
    sorted_indices = [index for value, index in sorted_list]
    return sorted_indices


def random_sampling(data_size, num_samples):
    return np.random.choice(data_size, num_samples, replace=False)


def get_eveloved_nodes(args, replay_num, evo_num):
    replay_list = []
    # should be N*T
    past_path = osp.join(args.raw_data_path, str(args.year-1)+'.npz')
    past_data = np.load(past_path)['x']
    daily_node_past = np.mean(past_data.reshape(-1, 288, past_data.shape[1]), axis=1).T
    
    current_path = osp.join(args.raw_data_path, str(args.year)+'.npz')
    current_data = np.load(current_path)['x']
    daily_node_cur = np.mean(current_data.reshape(-1, 288, current_data.shape[1]), axis=1).T
    
    if daily_node_past.shape[0] < daily_node_past.shape[1]:
        daily_node_cur = daily_node_cur.transpose(1, 0)
        daily_node_past = daily_node_past.transpose(1, 0)

    daily_node_cur = daily_node_cur[:daily_node_past.shape[0], :]

    distance = []
    if daily_node_past.shape[0] > daily_node_past.shape[1]:
        random_replay = random_sampling(daily_node_past.shape[0], int(replay_num))
        random_evo = random_sampling(daily_node_past.shape[0], evo_num)
        return random_replay, random_evo
    
    for i in range(daily_node_past.shape[0]):
        distance.append(WD(daily_node_past[i], daily_node_cur[i]))
    
    sorted_index = sort_with_index(distance)
    replay_node = sorted_index[-int(replay_num*0.1):]
    replay_list.extend(replay_node)
    evo_node = list(sorted_index[:evo_num])
    replay_sample = random_sampling(daily_node_past.shape[0], int(replay_num*0.9))
    replay_list.extend(replay_sample)
    
    return replay_list, evo_node 