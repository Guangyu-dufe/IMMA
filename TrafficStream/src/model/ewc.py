import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import numpy as np
import logging
import pdb

from torch_geometric.data import Data


class EWC(nn.Module):

    def __init__(self, model, adj, ewc_lambda = 0, ewc_type = 'ewc'):
        super(EWC, self).__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type
        self.adj = adj

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, loader, lossfunc, device):
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters() if param[1].requires_grad]
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        
        for i, data in enumerate(loader):
            data = data.to(device, non_blocking=True)
            data.y.requires_grad_(True)
            
            with torch.no_grad():
                self.model._momentum_update(False)
                
            if self.model.args.extra_feature:
                from types import SimpleNamespace
                basic_data = SimpleNamespace()
                basic_data.x = data.x.reshape(-1, self.model.args.gcn["in_channel"]*2)
            else:
                from types import SimpleNamespace
                basic_data = SimpleNamespace()
                basic_data.x = data.x.reshape(-1, self.model.args.gcn["in_channel"]*2)[:, :self.model.args.gcn["in_channel"]]
            basic_data.batch = data.batch

            # if self.model.args.expand:
            #     basic_features = self.model.feature(basic_data, self.adj)
            # else:
            basic_features = self.model.basic_model.feature(basic_data, self.adj)
            log_likelihood = lossfunc(data.y, basic_features, reduction='mean')
            
            grad_log_liklihood = autograd.grad(log_likelihood, trainable_params, 
                                             create_graph=True, 
                                             retain_graph=True,
                                             allow_unused=True)
            
            for name, grad in zip(_buff_param_names, grad_log_liklihood):
                if grad is not None:
                    est_fisher_info[name] += grad.data.clone() ** 2
                else:
                    est_fisher_info[name] += torch.zeros_like(trainable_params[_buff_param_names.index(name)].data)
                
        for name in _buff_param_names:
            self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])


    def register_ewc_params(self, loader, lossfunc, device):
        self._update_fisher_params(loader, lossfunc, device)
        self._update_mean_params()


    def compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            try:
                estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
                if estimated_fisher is None:
                    losses.append(0)
                elif self.ewc_type == 'l2':
                    losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
                else:
                    losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            except AttributeError as e:
                # logging.warning(f"Parameter {_buff_param_name} not found in EWC buffers. Skipping this parameter.")
                continue
        return 1 * (self.ewc_lambda / 2) * sum(losses)
    
    def forward(self, data, adj): 
        return self.model(data, adj)

