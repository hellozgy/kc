#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from dataset import Constants
import ipdb


class Inception(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert (co % 4 == 0)
        cos = [int(co // 4)] * 4

        self.activa = nn.Sequential()

        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))

        if relu:
            self.activa.add_module('relu', nn.Tanh())

        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[0], 1, stride=1))
        ]))

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))

        self.branch3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin, cos[2], 3, stride=1, padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1)),
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(torch.cat((branch1, branch2, branch3, branch4), 1))
        return result


class CNNInceptionText(nn.Module):
    def __init__(self, opt):
        super(CNNInceptionText, self).__init__()
        incept_dim = opt.inception_dim
        self.opt = opt
        self.encoder = nn.Embedding(opt.vocab_size, opt.embeds_size,
                                   padding_idx=Constants.PAD_INDEX)
        self.conv = nn.Sequential(
            Inception(opt.embeds_size, incept_dim),
            Inception(incept_dim, incept_dim),
            nn.MaxPool1d(opt.max_len)
        )

        self.fc = nn.Sequential(
            nn.Linear(incept_dim, opt.hidden_size),
            nn.BatchNorm1d(opt.hidden_size),
            nn.Tanh(),
            nn.Linear(opt.hidden_size, opt.num_classes),
            nn.Sigmoid()
        )

        if opt.embeds_path:
            print('load embedding')
            print('embedding path', opt.embeds_path)
            self.encoder.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))
    
    def forward(self, x):
        content = self.encoder(x)
        content = self.conv(content.permute(0, 2, 1))
        content = content.view(content.size(0), -1)
        out = self.fc(content)
        return out

    def get_optimizer(self, lr=1e-3, lr2=1e-6, weight_decay=0):
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr),
            {'params': self.encoder.parameters(), 'lr': lr2}
        ])
        return optimizer
  
