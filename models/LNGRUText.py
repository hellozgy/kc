from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX
from .layer_norm import LayerNorm
from .rnn import LNGRUCell, LNGRU
from .activate import Swish,Highway
import ipdb
import torch.nn.utils.rnn as rnn_util
from .func import swish

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class LNGRUText(BasicModule):
    def __init__(self, opt):
        super(LNGRUText, self).__init__()
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
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path[:-4]+'.npz')['vec']))
        self.gru = nn.ModuleList([nn.GRU(input_size=self.embeds_size, hidden_size=self.hidden_size, bidirectional=True)]+
                                  [nn.GRU(input_size=2*self.hidden_size, hidden_size=self.hidden_size, bidirectional=True) for _ in range(opt.num_layers-1)])
        # self.ln = nn.ModuleList([LayerNorm(self.embeds_size)]+
        #     [LayerNorm(2*self.hidden_size) for _ in range(opt.num_layers-1)])

        fc_size = self.hidden_size*2+opt.num_layers*2*self.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_size, self.linear_hidden_size),
            nn.BatchNorm1d(self.linear_hidden_size),
            Swish(),
            nn.Linear(self.linear_hidden_size, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, content):
        seq_len = content.size(1)
        content = self.embeds(content).permute(1, 0, 2)

        hiddens = []
        input = content
        # pre = 0
        for layer, gru in enumerate(self.gru):
            # input = ln(input)
            output, hn = gru(input)
            hiddens.append(torch.cat([hn[0],hn[1]], 1))
            input = output + (input if layer>0 else 0)

        hiddens = torch.cat(hiddens, 1)
        output = F.max_pool1d(output.permute(1,2,0), seq_len).squeeze(2)
        # output = torch.cat([hiddens, output], 1)
        predicts = self.fc(output)
        return predicts