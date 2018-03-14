#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX
from .modules import ConvBlock, Conv
from .activate import Swish
from .BasicModule import BasicModule

class DPCNNText(BasicModule):
    def __init__(self, opt, ):
        super(DPCNNText, self).__init__()
        assert opt.vocab_size > 0
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.embeds = nn.Embedding(opt.vocab_size, self.embeds_size, padding_idx=PAD_INDEX)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path[:-4] + '.npz')['vec']))
        # conv层
        self.conv1 = nn.Sequential(
            Conv(opt.embeds_size),
            Conv(opt.embeds_size),
        )
        self.conv2 = nn.ModuleList([
            ConvBlock(dim=opt.embeds_size, seq_len=int(opt.max_len/(2**i))) for i in range(opt.num_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(opt.embeds_size, opt.embeds_size),
            nn.BatchNorm1d(opt.embeds_size),
            Swish(),
            nn.Linear(opt.embeds_size, opt.num_classes),
            # nn.BatchNorm1d(opt.num_classes),
        )

    def forward(self, inputs, bw):
        x = self.embeds(inputs).permute(0, 2, 1)
        x1 = self.conv1(x)
        x2 = x1
        for conv in self.conv2:
            x2 = conv(x2)
        x3 = F.max_pool1d(x2, kernel_size=x2.size(-1)).squeeze()
        output = self.fc(x3)
        return output