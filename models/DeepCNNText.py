#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

        # conv层
        for i in range(len(self.filters)):
            conv = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filter_nums[i],
                             kernel_size=self.embeds_size * self.filters[i], stride=self.embeds_size)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Sequential(
            nn.Linear(sum(self.filter_nums), 2 * self.hidden_size),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 2, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.Sigmoid()
        )

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inputs):
        x = self.embeds(inputs).view(-1, 1, self.embeds_size * self.max_len)



        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)),
                         self.max_len - self.filters[i] + 1)
            for i in range(len(self.filters))
        ]

        x = torch.cat(conv_results, dim=1)
        x = F.dropout(x, p=self.dropout)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)
        return x

    def get_optimizer(self, lr=1e-3, lr2=0, weight_decay=0):
        ignored_params = list(map(id, list(map(id, self.embeds.parameters()))))

        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr),
            {'params': self.embeds.parameters(), 'lr': lr}
        ])
        return optimizer









