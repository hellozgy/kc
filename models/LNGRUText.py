from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX
from .layer_norm import LayerNorm
from .rnn import LNGRUCell,LNGRU
import ipdb
import torch.nn.utils.rnn as rnn_util


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

        self.gru = nn.ModuleList([LNGRU(LNGRUCell, input_size=self.embeds_size, hidden_size=self.hidden_size, bidirectory=True)]+
                                  [LNGRU(LNGRUCell, input_size=2*self.hidden_size, hidden_size=self.hidden_size, bidirectory=True) for _ in range(opt.num_layers-1)])
        self.ln = nn.ModuleList([LayerNorm(self.embeds_size)]+
            [LayerNorm(2*self.hidden_size) for _ in range(opt.num_layers-1)])

        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling*(self.hidden_size*2)+opt.num_layers*2*self.hidden_size, self.linear_hidden_size),
            LayerNorm(self.linear_hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.linear_hidden_size, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, content):
        content = self.embeds(content).permute(1, 0, 2)
        content = F.dropout(content, p=self.dropout, training=self.training)
        hiddens = []
        input = content
        pre = 0
        for layer, (gru, ln) in enumerate(zip(self.gru, self.ln)):
            input = F.sigmoid(ln(input))
            ipdb.set_trace()
            output, hn = gru(input)
            hiddens.append(torch.cat([hn[0],hn[1]], 1))
            input = output + pre
            pre = pre + output

        hiddens = torch.cat(hiddens, 1)
        content_lstm = output.permute(1, 2, 0)  # content_lstm: (batch, dim, seq_len)
        content_conv_out = kmax_pooling(content_lstm, 2, self.kmax_pooling)
        reshaped = content_conv_out.view(content_conv_out.size(0), -1)
        output = torch.cat([hiddens, reshaped], 1)
        predicts = self.fc(output)
        return predicts