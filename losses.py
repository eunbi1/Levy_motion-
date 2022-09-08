import torch
import copy
import time
import numpy as np
from scipy.special import gamma
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_fn(x):
    return torch.tensor(gamma(x))


def get_continuous_time(index):
    return index / 1000


def get_discrete_time(t, N):
    return t * N


def gamma_fn(x):
    return torch.tensor(gamma(x))

from torchlevy import LevyStable
levy = LevyStable()

def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_L: torch,
            num_steps=1000, type="backpropagation"):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)

    if sde.alpha == 2:
        score = -1 / 2 * (e_L)*torch.pow(sigma+1e-4,-1)[:,None,None,None]*sde.beta(t)[:,None,None,None]

    else:
        score = levy.score(e_L, sde.alpha, type=type).to(device)* torch.pow(sigma+1e-4,-1)[:,None,None,None]*sde.beta(t)[:,None,None,None]

    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]
    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    output = model(x_t, t)*sde.beta(t)[:,None,None,None]
    weight = (output - score)

    return (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)  # +torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)



