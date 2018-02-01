from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class LSTMText(BasicModule): 
    def __init__(self, opt):
        super(LSTMText, self).__init__(opt)
        self.vocab_size = opt.vocab_size
        assert self.vocab_size > 0
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.num_classes = opt.num_classes
        self.kmax_pooling = opt.kmax_pooling
        self.linear_hidden_size = opt.linear_hidden_size
        self.embeds = nn.Embedding(self.vocab_size, self.embeds_size, padding_idx=PAD_INDEX)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))

        self.lstm = nn.LSTM(input_size=self.embeds_size,
                            hidden_size=self.hidden_size,
                            num_layers=opt.num_layers,
                            bias=True,
                            batch_first=False,
                            bidirectional=True,
                            dropout=self.dropout
                            )

        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling*(self.hidden_size*2), self.linear_hidden_size),
            nn.BatchNorm1d(self.linear_hidden_size),
            nn.Tanh(),
            nn.Linear(self.linear_hidden_size, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, content):
        content = self.embeds(content)
        content = F.dropout(content, p=self.dropout, training=self.training)
        content_lstm = self.lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)  # content_lstm: (batch, dim, seq_len)
        content_conv_out = kmax_pooling(content_lstm, 2, self.kmax_pooling)
        reshaped = content_conv_out.view(content_conv_out.size(0), -1)
        predicts = self.fc(reshaped)
        return predicts