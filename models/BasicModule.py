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

    def GRU_FF(self, GRUCell, h0, inputs, lengths):
        '''
        :param GRUCell:
        :param h0:
        :param inputs:
        :param lengths:
        :return: output:seq_len *batch * hidden_size
                hn:batch * hidden_size
        '''
        seq_len, batch_size, hidden_size = inputs.size()




    def get_optimizer(self, lr=1e-3, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.embeds.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr),
            {'params': self.embeds.parameters(), 'lr': lr2}
        ])
        return optimizer

    def update_optimizer(self, optimizer,  lr, lr2):
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr2


