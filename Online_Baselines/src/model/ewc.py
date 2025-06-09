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
        # Initialize fisher_information and parameter_means to avoid AttributeError
        self.fisher_information = {}
        self.parameter_means = {}

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        model_state = self.model.state_dict()
        for key in model_state:
            state_dict[f'model.{key}'] = model_state[key]
            
        return state_dict
        
    def load_state_dict(self, state_dict, strict=True):
        model_state = {}
        ewc_state = {}
        
        for key in list(state_dict.keys()):
            if key.startswith('model.'):
                model_state[key[6:]] = state_dict.pop(key)
            else:
                ewc_state[key] = state_dict[key]
        
        self.model.load_state_dict(model_state, strict=False)
        
        return super().load_state_dict(ewc_state, strict=False)

    def count_parameters(self):
        return self.model.count_parameters()

    def _update_mean_params(self):
        # Initialize means for most recent task
        self.parameter_means = {}
        # Store current parameters as the means
        for n, p in self.model.named_parameters():
            self.parameter_means[n] = p.clone().detach()

    def _update_fisher_params(self, loader, lossfunc, device):
        # Initialize fisher information for most recent task
        self.fisher_information = {}
        # Initialize means for most recent task
        self.parameter_means = {}
        # Initialize precision matrices
        precision_matrices = {}
        
        # Initialize fisher information for each parameter
        for n, p in self.model.named_parameters():
            precision_matrices[n] = p.clone().detach().fill_(0).to(device)

        self.model.eval()
        for data in loader:
            self.model.zero_grad()
            data = data.to(device)
            output = self.model(data, self.adj)
            
            # Handle models that return tuples (e.g., STKEC returns pred, attention)
            if isinstance(output, tuple):
                pred = output[0]  # Use the first element as prediction
            else:
                pred = output
                
            loss = lossfunc(pred, data.y)
            log_likelihood = -loss

            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
            
            for (n, p), g in zip(self.model.named_parameters(), grad_log_liklihood):
                if g is not None:  # Only update precision matrix if gradient exists
                    precision_matrices[n].data += g.data ** 2

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.fisher_information = precision_matrices

        # Store current parameters as the means
        for n, p in self.model.named_parameters():
            self.parameter_means[n] = p.clone().detach()

    def _update_fisher_params_for_stkec(self, loader, lossfunc, device):
        # Initialize fisher information for most recent task (STKEC version)
        self.fisher_information = {}
        # Initialize precision matrices
        precision_matrices = {}
        
        # Initialize fisher information for each parameter
        for n, p in self.model.named_parameters():
            precision_matrices[n] = p.clone().detach().fill_(0).to(device)
        
        self.model.eval()
        for i, data in enumerate(loader):
            self.model.zero_grad()
            data = data.to(device, non_blocking=True)
            output = self.model.forward(data, self.adj)
            
            # Handle models that return tuples (e.g., STKEC returns pred, attention)
            if isinstance(output, tuple):
                pred = output[0]  # Use the first element as prediction
            else:
                pred = output
                
            log_likelihood = lossfunc(data.y, pred, reduction='mean')
            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
            
            for (n, p), g in zip(self.model.named_parameters(), grad_log_liklihood):
                if g is not None:  # Only update precision matrix if gradient exists
                    precision_matrices[n].data += g.data ** 2
        
        # Set fisher_information for compute_consolidation_loss to use
        self.fisher_information = precision_matrices

    def _update_fisher_params_for_pecpm(self, loader, lossfunc, device):
        # Initialize fisher information for most recent task (PECPM version)
        self.fisher_information = {}
        # Initialize precision matrices
        precision_matrices = {}
        
        # Initialize fisher information for each parameter
        for n, p in self.model.named_parameters():
            precision_matrices[n] = p.clone().detach().fill_(0).to(device)
        
        self.model.eval()
        for i, data in enumerate(loader):
            self.model.zero_grad()
            data = data.to(device, non_blocking=True)
            output = self.model.forward(data, self.adj)
            
            # Handle models that return tuples (e.g., PECPM returns pred, attention)
            if isinstance(output, tuple):
                pred = output[0]  # Use the first element as prediction
            else:
                pred = output
                
            log_likelihood = lossfunc(data.y, pred, reduction='mean')
            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)
            
            for (n, p), g in zip(self.model.named_parameters(), grad_log_liklihood):
                if g is not None:  # Only update precision matrix if gradient exists
                    precision_matrices[n].data += g.data ** 2
        
        # Set fisher_information for compute_consolidation_loss to use
        self.fisher_information = precision_matrices

    def register_ewc_params(self, loader, lossfunc, device):
        self._update_fisher_params(loader, lossfunc, device)
        self._update_mean_params()
    
    
    def register_ewc_params_for_stkec(self, loader, lossfunc, device):
        self._update_fisher_params_for_stkec(loader, lossfunc, device)
        self._update_mean_params()

    def register_ewc_params_for_pecpm(self, loader, lossfunc, device):
        self._update_fisher_params_for_pecpm(loader, lossfunc, device)
        self._update_mean_params()

    def compute_consolidation_loss(self):
        # Check if fisher_information and parameter_means have been initialized
        if not hasattr(self, 'fisher_information') or not hasattr(self, 'parameter_means'):
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        if not self.fisher_information or not self.parameter_means:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        losses = []
        for n, p in self.model.named_parameters():
            if n in self.fisher_information and n in self.parameter_means:
                if self.ewc_type == 'l2':
                    losses.append((10e-6 * (p - self.parameter_means[n]) ** 2).sum())
                else:
                    losses.append((self.fisher_information[n] * (p - self.parameter_means[n]) ** 2).sum())
        return 1 * (self.ewc_lambda / 2) * sum(losses) if losses else torch.tensor(0.0, device=p.device)
    
    def forward(self, data, adj): 
        return self.model(data, adj)

