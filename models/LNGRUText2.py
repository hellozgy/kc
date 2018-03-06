from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX
from .layer_norm import LayerNorm
from torch.autograd import Variable
from .rnn import LNGRUCell, LNGRU
from .activate import Swish,Highway
import ipdb
import torch.nn.utils.rnn as rnn_util
from .func import swish

class LNGRUText2(BasicModule):
    def __init__(self, opt):
        super(LNGRUText2, self).__init__()
        assert opt.vocab_size > 0
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.embeds = nn.Embedding(opt.vocab_size, self.embeds_size, padding_idx=PAD_INDEX)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path[:-4]+'.npz')['vec']))
        self.gru = LNGRU(input_size=self.embeds_size, hidden_size=self.hidden_size, bidirectional=True)
        self.h0 = self.init_hidden((2, 1, self.hidden_size))
        self.ln = LayerNorm(self.embeds_size)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 4, opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            Swish(),
            nn.Linear(opt.linear_hidden_size, opt.num_classes),
        )
    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal(h0)
        return h0

    def forward(self, content, bw):
        batch_size = content.size(0)
        seq_len = content.size(1)
        content = self.embeds(content).permute(1, 0, 2)
        input = self.ln(content)
        output, hn = self.gru(input, self.h0.repeat(1, batch_size, 1))
        hiddens = hn.transpose(0, 1).contiguous().view(batch_size, -1)
        output = F.max_pool1d(output.permute(1, 2, 0), seq_len).squeeze(2)
        output = torch.cat([hiddens, output], 1)
        predicts = self.fc(output)
        return predicts