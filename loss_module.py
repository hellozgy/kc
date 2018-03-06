import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb

class NLLLoss6(nn.Module):
    def __init__(self,  weight=[[1,1] for _ in range(6)], size_average=True):
        super(NLLLoss6, self).__init__()
        self.weight = torch.FloatTensor(weight).cuda()
        self.losses = [nn.NLLLoss(self.weight[i], size_average=True) for i in range(6)]

    def update_weight(self, weight):
        self.weight = torch.FloatTensor(weight).cuda()
        self.losses = [nn.NLLLoss(self.weight[i], size_average=True) for i in range(6)]

    def forward(self, predicts, targets):
        '''
        :param predicts: (batch, 6)
        :param targets: (batch, 6)
        :return:
        '''
        loss = 0
        predicts = F.sigmoid(predicts)
        for i in range(6):
            predict = torch.cat([1-predicts[:, i:i+1], predicts[:, i:i+1]], 1)
            predict = torch.log(predict)
            target = targets[:, i].long()
            loss = loss + self.losses[i](predict, target)
        return loss

if __name__=='__main__':
    torch.cuda.set_device(3)
    predict = Variable(torch.randn(10, 6)).cuda()
    target = Variable(torch.ones(10,6)).cuda()
    loss = NLLLoss6()
    print(loss(predict, target))
