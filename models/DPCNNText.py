import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset.Constants import PAD_INDEX
from .modules import ConvBlock, Conv
from .activate import Swish
from .BasicModule import BasicModule
from .layer_norm import LayerNorm

class DPCNNText(BasicModule):
    def __init__(self, opt, ):
        super(DPCNNText, self).__init__()
        assert opt.vocab_size > 0
        self.layers = opt.num_layers
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

        self.ln = nn.ModuleList([LayerNorm(opt.embeds_size, dim=1)
                                 for _ in range(opt.num_layers+1)])

        self.fc = nn.Sequential(
            self.Linear(opt.embeds_size, opt.linear_hidden_size),
            LayerNorm(opt.linear_hidden_size),
            Swish(),
            self.Linear(opt.linear_hidden_size, opt.num_classes),
        )

    def forward(self, inputs, bw):
        x = self.embeds(inputs).permute(0, 2, 1)
        x = self.ln[0](x)
        x1 = self.conv1(x)
        x2 = x1
        for layer, conv in enumerate(self.conv2):
            x2 = conv(x2)
        x3 = F.max_pool1d(x2, kernel_size=x2.size(-1)).squeeze()
        output = self.fc(x3)
        return output