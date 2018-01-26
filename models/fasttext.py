#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict


class FastText(nn.Module):
    """pytorch实现的简单fasttext分类"""
    def __init__(self, opt):
        super(FastText, self).__init__()
        self.opt = opt
        self.model_name = 'FastText'
        self.embeds_size = opt.embeds_size
        self.embeds = nn.Embedding(opt.vocab_size, opt.embeds_size)
        self.hidden_size = opt.hidden_size
        self.num_classes = opt.num_classes

        self.fc = nn.Sequential(
            nn.Linear(self.embeds_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.Sigmoid()
        )

        if opt.embeds_path:
            print('load embedding')
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))

    def forward(self, content):
        embeds_ctx = self.embeds(content)
        content = torch.mean(embeds_ctx, 1)
        content = content.view(content.size(0), -1)
        output = self.fc(content)
        return output

    def get_optimizer(self, lr=1e-3, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.embeds.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr),
            {'params': self.embeds.parameters(), 'lr': lr2}
        ])
        return optimizer