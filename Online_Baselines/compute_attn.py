import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

def extract_representative_patterns(data, K=32, window_size=12):
    T, N = data.shape
    
    daily_steps = 288
    num_days = T // daily_steps
    
    if T >= daily_steps:
        complete_days_data = data[:num_days * daily_steps]
        daily_data = complete_days_data.reshape(num_days, daily_steps, N)
        daily_avg_data = np.mean(daily_data, axis=1)
    else:
        daily_avg_data = data
        num_days = T
    
    patterns_raw = []
    for i in range(N):
        for t in range(num_days - window_size + 1):
            window = daily_avg_data[t:t+window_size, i]
            patterns_raw.append(window)
    
    patterns_raw = np.array(patterns_raw)
    
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(patterns_raw)
    
    patterns = kmeans.cluster_centers_
    
    return patterns

def split_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    T = data.shape[0]
    
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def compute_matching_degree_matrix(input_sequences, patterns, k_c=10):
    N, T_h = input_sequences.shape
    K, _ = patterns.shape
    
    similarity_matrix = cosine_similarity(input_sequences, patterns)
    
    Q = np.zeros_like(similarity_matrix)
    for i in range(N):
        top_k_indices = np.argsort(similarity_matrix[i])[-k_c:]
        Q[i, top_k_indices] = similarity_matrix[i, top_k_indices]
    
    return Q

def compute_attention_scores(spatial_temporal_representation, pattern_bank):
    similarity = cosine_similarity(spatial_temporal_representation, pattern_bank)
    
    P = F.softmax(torch.tensor(similarity), dim=1).numpy()
    
    return P

def compute_attention_labels_for_split(data_split, patterns, K=32, window_size=12, k_c=10):
    T, N = data_split.shape
    
    daily_steps = 288
    num_days = T // daily_steps
    
    if T >= daily_steps:
        complete_days_data = data_split[:num_days * daily_steps]
        daily_data = complete_days_data.reshape(num_days, daily_steps, N)
        daily_avg_data = np.mean(daily_data, axis=1)
        
        Q_daily = []
        for t in range(num_days - window_size + 1):
            current_window = daily_avg_data[t:t+window_size, :].T
            
            Q_t = compute_matching_degree_matrix(current_window, patterns, k_c)
            Q_daily.append(Q_t)
        
        Q_daily = np.array(Q_daily)
        
        Q_all = np.zeros((T, N, K))
        
        for day_idx in range(len(Q_daily)):
            start_step = day_idx * daily_steps
            end_step = min((day_idx + 1) * daily_steps, T)
            Q_all[start_step:end_step, :, :] = Q_daily[day_idx]
        
        if len(Q_daily) > 0:
            for t in range(min(window_size-1, T)):
                Q_all[t, :, :] = Q_daily[0]
        
        if len(Q_daily) > 0:
            last_valid_day = len(Q_daily) - 1
            remaining_start = last_valid_day * daily_steps + daily_steps
            if remaining_start < T:
                Q_all[remaining_start:, :, :] = Q_daily[last_valid_day]
        
    else:
        Q_all = np.zeros((T, N, K))
        
        Q_valid = []
        for t in range(T - window_size + 1):
            current_window = data_split[t:t+window_size, :].T
            
            Q_t = compute_matching_degree_matrix(current_window, patterns, k_c)
            Q_valid.append(Q_t)
        
        Q_valid = np.array(Q_valid)
        
        for t in range(window_size-1):
            Q_all[t, :, :] = Q_valid[0] if len(Q_valid) > 0 else 0
        
        for t in range(len(Q_valid)):
            Q_all[t + window_size - 1, :, :] = Q_valid[t]
    
    return Q_all

def compute_attention_labels(data, K=32, window_size=12, k_c=10, train_ratio=0.6):
    T, N = data.shape
    
    train_end = int(T * train_ratio)
    train_data = data[:train_end]
    
    patterns = extract_representative_patterns(train_data, K, window_size)
    
    Q_all = compute_attention_labels_for_split(data, patterns, K, window_size, k_c)
    
    return Q_all, patterns

if __name__ == "__main__":
    for year in range(2017,2021):
        print(f"Processing year {year}")
        import os
        import os.path as osp
        data = np.load(osp.join('/home/bd2/ANATS/Oline_Baselines/PECPM/data/SD/finaldata',str(year)+'.npz'))['x']
        T, N = data.shape
        
        print(f"input data shape: {data.shape}")
        
        Q, patterns = compute_attention_labels(data, K=32, window_size=12, train_ratio=0.6)
        att_label = Q.reshape(-1,32)
        att_label = np.nan_to_num(att_label)
        os.makedirs('data/SD/attetion', exist_ok=True)
        print(f"attention label shape: {att_label.shape}")

        np.save(f'data/SD/attetion/{year}_attention.npy', att_label)
        print(f"saved attention label for year {year}")
