import torch
import torch.nn as nn
from .activate import Swish
from torch.autograd import Variable
import ipdb

class Conv(nn.Module):
    def __init__(self, dim, activate=Swish()):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels=dim,
                              out_channels=dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.activate = activate

    def forward(self, x):
        x1 = self.activate(x)
        x2 = self.conv(x1)
        return x2

class ConvBlock(nn.Module):
    def __init__(self, dim, seq_len, activate=Swish()):
        super(ConvBlock, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1 if seq_len%2==0 else 0)
        self.conv = nn.Sequential(
            Conv(dim, activate=activate),
            Conv(dim, activate=activate)
        )


    def forward(self, x):
        x1 = self.pooling(x)
        x2 = self.conv(x1)
        x3 = x1 + x2
        return x3

if __name__=='__main__':
    torch.cuda.set_device(0)
    x = Variable(torch.randn(4, 10, 200)).cuda() # batch dim seq_len
    m = nn.ModuleList(
        [ConvBlock(dim=10, seq_len=int(200/(2**i))).cuda() for i in range(5)]
    )
    for mm in m:
        x = mm(x)
