#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .layer_norm import LayerNorm
from .activate import Swish
from dataset.Constants import PAD_INDEX
from .BasicModule import BasicModule
import ipdb


class InceptionText(BasicModule):
    def __init__(self, opt, ):
        super(InceptionText, self).__init__()
        assert opt.vocab_size > 0
        self.num_layers = opt.num_layers
        self.dropout = opt.dropout
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.embeds = nn.Embedding(opt.vocab_size, self.embeds_size, padding_idx=PAD_INDEX)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path[:-4] + '.npz')['vec']))

        convs = [nn.Sequential(
            self.Conv1d(in_channels=self.embeds_size,
                      out_channels=opt.filter_nums[i],
                      kernel_size=opt.filters[i]),
            nn.BatchNorm1d(opt.filter_nums[i]),
            Swish(),

            self.Conv1d(in_channels=opt.filter_nums[i],
                      out_channels=opt.filter_nums[i],
                      kernel_size=opt.filters[i]),
            nn.BatchNorm1d(opt.filter_nums[i]),
            Swish(),

            nn.MaxPool1d(kernel_size=opt.max_len - opt.filters[i] * 2 + 2)
        ) for i in range(len(opt.filters))]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            self.Linear(sum(opt.filter_nums),
                      2 * self.hidden_size),
            LayerNorm(self.hidden_size * 2),
            Swish(),
            self.Linear(self.hidden_size * 2, opt.num_classes),
        )


    def forward(self, inputs, bw):
        x = self.embeds(inputs).permute(0, 2, 1)
        conv_results = [conv(x) for conv in self.convs]

        x = torch.cat(conv_results, dim=1)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x