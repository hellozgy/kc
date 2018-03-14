from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX
from .layer_norm import LayerNorm
from .activate import Swish
import ipdb

class LNGRUText(BasicModule):
    def __init__(self, opt):
        super(LNGRUText, self).__init__()
        assert opt.vocab_size > 0
        self.num_layers = opt.num_layers
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.embeds = nn.Embedding(opt.vocab_size, self.embeds_size, padding_idx=PAD_INDEX)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path[:-4]+'.npz')['vec']))

        self.ln = nn.ModuleList([LayerNorm(self.embeds_size)]+[LayerNorm(self.hidden_size*2) for _ in range(opt.num_layers-1)])  # better than bn
        self.h0 = self.init_hidden((2*opt.num_layers, 1, opt.hidden_size))
        self.gru = nn.ModuleList([self.GRU(input_size=self.embeds_size, hidden_size=self.hidden_size, bidirectional=True)]+\
                   [self.GRU(input_size=self.hidden_size*2, hidden_size=self.hidden_size, bidirectional=True)
                    for _ in range(opt.num_layers-1)])
        self.fc = nn.Sequential(
            self.Linear(self.hidden_size*2 + self.hidden_size*2*opt.num_layers, opt.linear_hidden_size),
            LayerNorm(opt.linear_hidden_size), # layer norm is better than bn
            Swish(),
            self.Linear(opt.linear_hidden_size, opt.num_classes),
        )


    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal(h0)
        return h0

    def forward(self, content, bw):
        batch_size = content.size(0)
        seq_len = content.size(1)
        content = self.embeds(content).permute(1, 0, 2) #no dropout is better]
        input = content
        hiddens = []
        for layer in range(self.num_layers):
            input = self.ln[layer](input)
            output, hn = self.gru[layer](input, self.h0[2*layer:2*(layer+1)].repeat(1, batch_size, 1))
            hiddens.append(hn.transpose(0, 1).contiguous().view(batch_size, -1))
            input = output

        output = F.max_pool1d(output.permute(1,2,0), seq_len).squeeze(2)
        output = torch.cat(hiddens+[output], 1)
        predicts = self.fc(output)
        return predicts