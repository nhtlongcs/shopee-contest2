import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return self.loss(output, target)

class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, beta, **kwargs):
        if isinstance(beta, (float,int)):
            self.beta = torch.Tensor([beta])
        if isinstance(beta, list):
            self.beta = torch.Tensor(beta)
        super().__init__(pos_weight=self.beta, **kwargs)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__() 

    def forward(self, input, target):
        return 0
        