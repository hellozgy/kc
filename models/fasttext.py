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
        self.model_name = 'fasttext'
        self.embeds = nn.Embedding(opt.vocab_size, opt.embeds_size)

        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.Sigmoid()
        )

        if opt.embedding_path:
            print('load embedding')
            self.encoder.weight.data.copy_(torch.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, content):
        content = torch.mean(self.embeds(content), dim=1)
        output = self.fc(content)
        return output

    def get_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer