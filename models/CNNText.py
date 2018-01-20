#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
        class_num: int 类别数目
        filters: list 表示卷及层的filter大小，他的长度也表示模型有多少卷基层
        filter_nums: list filter个数
    """
    def __init__(self, max_sen_len, word_dim, vocab_size,
                 class_num, filters, filter_nums, wv_matrix,
                 batch_size=32,
                 model_type='multichannel', dropout_p=0.5):
        super(CNNText, self).__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_seq_len = max_sen_len
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.filters = filters
        self.filter_nums = filter_nums
        self.in_channel = 1
        self.dropout_p = dropout_p

        assert (len(self.filters) == len(self.filter_nums))

        self.embedding = nn.Embedding(self.vocab_size, self.word_dim)
        # 加载现有的word2vec模型
        self.embedding.weight.data.copy_(torch.from_numpy(wv_matrix))

        if self.model_type == 'static' or 'multichannel':
            self.embedding.weight.requires_grad = False

        self.embedding2 = None

        if self.model_type == 'multichannel':
            self.in_channel = 2
            self.embedding2 = nn.Embedding(self.vocab_size, self.word_dim)
            self.embedding2.weight.data.copy_(torch.from_numpy(wv_matrix))

        # conv层
        for i in range(len(self.filters)):
            conv = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filter_nums[i],
                             kernel_size=self.word_dim * self.filters[i], stride=self.word_dim)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Sequential(
            nn.Linear(sum(self.filter_nums), self.class_num),
            nn.BatchNorm1d(self.class_num),
            nn.Sigmoid()
        )

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inputs):
        x = self.embedding(inputs).view(-1, 1, self.word_dim * self.max_seq_len)
        if self.model_type == 'multichannel':
            x2 = self.embedding2(inputs).view(-1, 1, self.word_dim * self.max_seq_len)
            x = torch.cat((x, x2), dim=1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)),
                         self.max_seq_len - self.filters[i] + 1)
            for i in range(len(self.filters))
        ]

        x = torch.cat(conv_results, dim=1)
        x = F.dropout(x, p=self.dropout_p)
        x = self.fc(x)

        return x







