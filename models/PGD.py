import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 2]:
        raise ValueError("norm must be np.inf, or 2.")
    reduc_ind = list(range(1, len(eta.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    elif norm == 2:
        norm = torch.sqrt(
            torch.max(
                avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
            )
        )
        factor = torch.min(torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm)
        eta *= factor
    return eta

def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf and L2 norms are " "currently implemented."
        )
    # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()


class Adversarial(nn.Module):
    def __init__(self, mode_robust, model_natural, cfg):
        super(Adversarial, self).__init__()
        self.norm = np.inf if cfg['norm'] == "np.inf" else int(cfg['norm'])
        self.perturb_steps = cfg['perturb_steps']
        self.epsilon = cfg['epsilon']
        self.step_size = cfg['step_size']
        # Create model
        self.model_robust = mode_robust
        self.model_natural = model_natural
        # define KL-loss
        self. criterion_kl = nn.KLDivLoss(reduction='sum')

    def forward(self, x_natural):
        self.model_robust.eval()
        self.model_natural.eval()

        # generate adversarial example
        eta = torch.zeros_like(x_natural)
        eta = clip_eta(eta, self.norm, self.epsilon)
        x_adv = x_natural.detach() + eta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss_kl = self.criterion_kl(F.log_softmax(self.model_robust(x_adv), dim=1),
                                            F.softmax(self.model_robust(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            optimal_perturbation = optimize_linear(grad, self.step_size, self.norm)
            x_adv = x_adv.detach() + optimal_perturbation
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            eta_x_adv = x_adv - x_natural
            eta_x_adv = clip_eta(eta_x_adv, self.norm, self.epsilon)
            x_adv = x_natural + eta_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = Variable(x_adv, requires_grad=False)
        return x_adv
            
def generate_adversarial(model_robust, x_natural, cfg):
    norm = np.inf if cfg['norm'] == "np.inf" else int(cfg['norm'])
    perturb_steps = cfg['perturb_steps']
    epsilon = cfg['epsilon']
    step_size = cfg['step_size']
    criterion_kl = nn.KLDivLoss(size_average=False)

    model_robust.eval()
    # generate adversarial example
    # eta = torch.zeros_like(x_natural)
    eta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    eta = clip_eta(eta, norm, epsilon)
    x_adv = x_natural.detach() + eta
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model_robust(x_adv), dim=1),
                                        F.softmax(model_robust(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        optimal_perturbation = optimize_linear(grad, step_size, norm)
        x_adv = x_adv.detach() + optimal_perturbation
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        eta_x_adv = x_adv - x_natural
        eta_x_adv = clip_eta(eta_x_adv, norm, epsilon)
        x_adv = x_natural + eta_x_adv
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv
