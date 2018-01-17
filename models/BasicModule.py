import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from dataset import Constants
import ipdb


class BasicModule(nn.Module):
    def __init__(self, opt):
        super(BasicModule, self).__init__()
        self.vocab_size = opt.vocab_size
        assert self.vocab_size > 0
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.num_classes = opt.num_classes
        self.embeds = nn.Embedding(self.vocab_size, self.embeds_size)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))
        self.encoder = nn.LSTM(self.embeds_size, self.hidden_size, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, content):
        embeds_ctx = self.embeds(content).permute(1, 0, 2)
        _, (hn, cn) = self.encoder(embeds_ctx)
        output = torch.cat([hn[0], hn[1]], 1)
        output = self.fc(output)
        return output

    def get_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

