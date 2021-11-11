import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class LBGATLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(LBGATLoss, self).__init__()
        self.beta = beta
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, out_adv, out_natural, out_orig, y):  
        # compute simultaneously loss(self.model_natural(x_natural), y) and
        # Loss(self.model_robust(x_adv), self.model_natural(x_natural))
        loss_mse = self.ce(out_orig, y) + self.mse(out_orig, out_adv)
        loss_kl = (1.0 / out_orig.size(0)) * \
            self.criterion_kl(F.log_softmax(out_adv, dim=1),\
                            F.softmax(out_natural, dim=1
            )
        )
        loss = loss_mse + self.beta * loss_kl
        return loss


