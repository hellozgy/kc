import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from dataset import Constants
import math
import torch.nn as nn
import ipdb


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def Linear(self, in_features, out_features):
        m = nn.Linear(in_features, out_features)
        nn.init.xavier_normal(m.weight.data)
        nn.init.uniform(m.bias.data, -0.02, 0.02)
        return m

    def GRU(self, input_size, hidden_size, **kwargs):
        m = nn.GRU(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal(param.data)
            elif 'bias' in name:
                nn.init.uniform(param.data, -0.02, 0.02)
        return m

    def Conv1d(self, in_channels, out_channels, kernel_size):
        """Weight-normalized Conv1d layer"""
        m = nn.Conv1d(in_channels, out_channels, kernel_size)
        std = math.sqrt(4 / (kernel_size * in_channels))
        m.weight.data.normal_(mean=0, std=std)
        m.bias.data.zero_()
        return nn.utils.weight_norm(m, dim=2)


    def get_optimizer(self, lr=1e-3, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.embeds.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr, amsgrad=True),
            dict(params=self.embeds.parameters(), weight_decay=weight_decay, lr=lr2, amsgrad=True)
        ])
        return optimizer

    # def get_optimizer_sgd(self, lr=1e-3, lr2=0, weight_decay=0):
    #     ignored_params = list(map(id, self.embeds.parameters()))
    #     base_params = filter(lambda p: id(p) not in ignored_params,
    #                          self.parameters())
    #     optimizer = torch.optim.SGD([
    #         dict(params=base_params, weight_decay=weight_decay, lr=lr),
    #         {'params': self.embeds.parameters(), 'lr': lr2}
    #     ])
    #     return optimizer



