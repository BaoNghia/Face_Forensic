import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class Attacks(nn.Module):
    def __init__(self, model, config):
        super(Attacks, self).__init__()
        self.norm = np.inf if config['norm'] == "np.inf" else int(config['norm'])
        self.perturb_steps = config['perturb_steps']
        self.epsilon = config['epsilon']
        self.step_size = config['step_size']
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.model = model

    def perturb_PGD(self, x_natural, targets):
        self.model.eval()

        ## generate adversarial example
        eta = torch.zeros_like(x_natural).uniform_(-self.epsilon, self.epsilon)
        eta = clip_eta(eta, self.norm, self.epsilon)
        x_adv = x_natural.detach() + eta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                            F.softmax(self.model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            optimal_perturbation = optimize_linear(grad, self.step_size, self.norm)
            x_adv = x_adv.detach() + optimal_perturbation
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)
            eta_x_adv = x_adv - x_natural
            eta_x_adv = clip_eta(eta_x_adv, self.norm, self.epsilon)
            x_adv = x_natural + eta_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(x_adv, requires_grad=False)
        return x_adv

    def perturb_TRADES(self, x_natural, targets):
        self.model.eval()
        batch_size = len(x_natural)
        eta = 0.001 * torch.randn_like(x_natural).detach()
        x_adv = x_natural.detach() + eta
        if self.norm == np.inf:
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = self.criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                                F.softmax(self.model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        elif self.norm == 2:
            delta = 0.001 * torch.randn_like(x_natural).detach()
            delta = Variable(delta.data, requires_grad=True)
            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr= self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta
                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * self.criterion_kl(F.log_softmax(self.model(adv), dim=1),
                                                    F.softmax(self.model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = x_natural + delta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return Variable(x_adv, requires_grad=False)

def generate_PGD(model, x_natural, cfg):
    norm = np.inf if cfg['norm'] == "np.inf" else int(cfg['norm'])
    perturb_steps = cfg['perturb_steps']
    epsilon = cfg['epsilon']
    step_size = cfg['step_size']
    criterion_kl = nn.KLDivLoss(reduction='none')
    model.eval()
    ## generate adversarial example
    eta = torch.zeros_like(x_natural).uniform_(-epsilon, epsilon)
    eta = clip_eta(eta, norm, epsilon)
    x_adv = x_natural.detach() + eta
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1))
        x_adv_grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        optimal_perturbation = optimize_linear(x_adv_grad, step_size, norm)
        x_adv = x_adv.detach() + optimal_perturbation
        # x_adv = torch.clamp(x_adv, 0.0, 1.0)
        eta_x_adv = x_adv - x_natural
        eta_x_adv = clip_eta(eta_x_adv, norm, epsilon)
        x_adv = x_natural + eta_x_adv
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv

def generate_TRADES(model, x_natural, cfg):
    norm = np.inf if cfg['norm'] == "np.inf" else int(cfg['norm'])
    perturb_steps = cfg['perturb_steps']
    epsilon = cfg['epsilon']
    step_size = cfg['step_size']
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = len(x_natural)
    model.eval()
    ## generate adversarial example
    eta = 0.001 * torch.randn_like(x_natural).detach()
    x_adv = x_natural.detach() + eta
    if norm == np.inf:
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    elif norm == 2:
        delta = 0.001 * torch.randn_like(x_natural).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = x_natural + delta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return Variable(x_adv, requires_grad=False)
    