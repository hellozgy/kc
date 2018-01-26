#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class CNNText(nn.Module):
    """
    http://www.aclweb.org/anthology/D14-1181
    参考这篇论文的做法，设计出一个双通道的TextCNN模型
    但是做一点改进，添加batchnorm等
    Args:
        model_type: 模型类别，static, non-static, multichannel
        max_sen_len: 序列最大长度
        word_dim: word vector dim
        vocab_size:
        num_classes: int 类别数目
        filters: list 表示卷及层的filter大小，他的长度也表示模型有多少卷基层
        filter_nums: list filter个数
    """
    def __init__(self, opt, ):
        super(CNNText, self).__init__()
        self.model_type = opt.model_type
        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.embeds_size = opt.embeds_size
        self.vocab_size = opt.vocab_size
        self.num_classes = opt.num_classes
        self.filters = opt.filters
        self.filter_nums = opt.filter_nums
        self.in_channel = 1
        self.dropout = opt.dropout
        self.hidden_size = opt.hidden_size

        assert (len(self.filters) == len(self.filter_nums))

        self.embeds = nn.Embedding(self.vocab_size, self.embeds_size)
        # 加载现有的word2vec模型
        print('load embedding')
        self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))

        if self.model_type == 'static' or 'multichannel':
            self.embeds.weight.requires_grad = False

        self.embedd2 = None

        if self.model_type == 'multichannel':
            self.in_channel = 2
            print('loading embedding 2')
            self.embedd2 = nn.Embedding(self.vocab_size, self.embeds_size)
            self.embedd2.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))

        convs = [nn.Sequential(
            nn.Conv1d(in_channels=self.embeds_size * self.in_channel,
                      out_channels=self.filter_nums[i],
                      kernel_size=self.filters[i]),
            nn.BatchNorm1d(self.filter_nums[i]),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=self.filter_nums[i],
                      out_channels=self.filter_nums[i],
                      kernel_size=self.filters[i]),
            nn.BatchNorm1d(self.filter_nums[i]),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=self.max_len - self.filters[i] * 2 + 2)
        ) for i in range(len(self.filters))]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(sum(self.filter_nums),
                      2 * self.hidden_size),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 2, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.Sigmoid()
        )

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inputs):
        x = self.embeds(inputs).permute(0, 2, 1)
        if self.model_type == 'multichannel':
            x2 = self.embedd2(inputs).permute(0, 2, 1)
            x = torch.cat((x, x2), dim=1)

        # ipdb.set_trace()

        conv_results = [conv(x) for conv in self.convs]

        x = torch.cat(conv_results, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_optimizer(self, lr=1e-3, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.embedd2.parameters())) + list(map(id, self.embeds.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr),
            {'params': self.embedd2.parameters(), 'lr': lr}
        ])
        return optimizer









