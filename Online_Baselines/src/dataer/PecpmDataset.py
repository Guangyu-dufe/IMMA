import numpy as np
import torch 
from torch_geometric.data import Data, Dataset

class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', att='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] # [T, Len, N]
            self.y = inputs[split+'_y'] # [T, Len, N]
            self.att = att # [T, N, C]
        else:
            self.x = x
            self.y = y
            self.att = att if att is not None else np.zeros((x.shape[0], x.shape[1], 32))
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)  # 转置：[Len, N] -> [N, Len]
        y = torch.Tensor(self.y[index].T)  # 转置：[Len, N] -> [N, Len]

        att = torch.Tensor(self.att[index])

        return Data(x=x, y=y, att=att)
    
class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs # [T, Len, N]
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return Data(x=x) 