import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(labels, num_classes):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:])
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LBGATLoss(nn.Module):
    def __init__(self, weight = None, alpha=1.0, **kwargs):
        super(LBGATLoss, self).__init__()
        self.alpha = alpha
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.mse = nn.MSELoss()
        # self.fl = FocalLoss(weight = weight, gamma=2)
        self.fl = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, out_adv, out_natural, out_orig, y):
        loss_mse = self.fl(out_orig, y) + self.mse(out_orig, out_adv)
        loss_kl = (1.0 / out_orig.size(0)) * \
            self.criterion_kl(
                F.log_softmax(out_adv, dim=1), 
                F.softmax(out_natural, dim=1)
            )
        loss = loss_mse + self.alpha * loss_kl
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        if self.weight != None:
            self.weight = torch.Tensor(self.weight).to(device)   # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
