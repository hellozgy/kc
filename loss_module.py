import torch
from torch import nn
from torch.autograd import Variable

class BCELossWeight(nn.Module):
    def __init__(self, ngpu, weight=[0.095844, 0.009996, 0.052948, 0.002996, 0.049364, 0.008805]):
        super(BCELossWeight, self).__init__()
        self.ngpu = ngpu
        weight = list(map(lambda x: (1-x)/x, weight))
        self.weight = Variable(torch.FloatTensor(weight).cuda(ngpu))

    def reset(self):
        self.weight = Variable(torch.FloatTensor([1, 1, 1, 1, 1, 1]).cuda(self.ngpu))

    def forward(self, predict, target):
        '''
        :param predict: (batch, 6)
        :param target: (batch, 6)
        :return:
        '''
        loss = -torch.mean(target*torch.log(predict) * self.weight + (1-target)*torch.log(1-predict))
        return loss
