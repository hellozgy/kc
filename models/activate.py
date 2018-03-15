from torch import nn
import torch
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

class Highway(nn.Module):
    def __init__(self, features, activate=F.tanh):
        super(Highway, self).__init__()
        self.features = features
        self.activate = activate
        self.W = nn.Linear(features, 2*features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.W.weight.data, -0.02, 0.02)

    def forward(self, x):
        assert x.dim()==2 and x.size(1) == self.features
        wx = self.W(x)
        h, t = torch.split(wx, self.features, -1)
        h, t = self.activate(h), F.sigmoid(t)
        return h * t + x * (1 - t)