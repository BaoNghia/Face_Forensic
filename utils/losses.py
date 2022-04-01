import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, Sequence
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def one_hot(labels, num_classes):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:])
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.,
                 smoothing: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        if alpha:
            self.alpha = torch.tensor(alpha, dtype=torch.float, device=device)
        else:
            self.alpha = None
        self.gamma = torch.tensor(gamma, dtype=torch.float, device=device)
        self.smoothing = torch.tensor(smoothing, dtype=torch.float, device=device)
        self.reduction = reduction
        self.ignore_index = ignore_index

        self.nll_loss = nn.NLLLoss(
            weight= self.alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        log_pt = log_p.gather(dim=-1, index=y.unsqueeze(1)).squeeze(1)

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class ImbalanceFocalLoss(nn.Module):
    def __init__(self,
                beta: float = 0.9999,
                gamma: float = 2.,
                reduction: str = 'mean',
                ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.beta =  beta
        self.gamma = torch.tensor(gamma, dtype=torch.float, device=device)
        self.reduction = reduction
        self.ignore_index = ignore_index


    def compute_weights(self, x, y):
        x_np =  x.cpu().detach().numpy()
        no_of_classes = x_np.shape[-1]
        values, samples_per_cls = np.unique(x_np, return_counts=True)
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float().to(device)

        labels_one_hot = F.one_hot(y, no_of_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        return weights, labels_one_hot

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        weights, labels_one_hot = self.compute_weights(x, y)
        BCLoss = F.binary_cross_entropy_with_logits(input = x, target = labels_one_hot, reduction = "none")
 
        modulator = torch.exp(-self.gamma * labels_one_hot * x \
            - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))

        loss = modulator * BCLoss
        focal_loss = weights * loss

        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()

        return loss


class LBGATLoss(nn.Module):
    def __init__(self, weight = None, beta=1.0, **kwargs):
        super(LBGATLoss, self).__init__()
        self.beta = beta
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.mse = nn.MSELoss()
        self.fl = FocalLoss(alpha = weight)
        self.ce = nn.CrossEntropyLoss()

    
    def forward(self, out_adv, out_natural, out_orig, y):
        features_adv, x4s_adv, logits_adv = out_adv
        features_natural, x4s_natural, logits_natural = out_natural
        features_orig, x4s_orig, logits_orig = out_orig
        batch_size = y.size(0)
        
        # loss_ce = self.ce(logits_natural, y)
        loss_ce = self.fl(logits_adv, y)
        loss_map = F.l1_loss(features_adv, features_orig)
        loss_kl = (1.0 / batch_size) * self.criterion_kl(F.log_softmax(logits_adv, dim=1), 
                                                        F.softmax(logits_natural, dim=1))
        loss = loss_ce + loss_map + self.beta * loss_kl
        return loss
