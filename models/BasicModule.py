import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from dataset import Constants
import ipdb


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

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


