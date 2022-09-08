from torchvision.utils import make_grid
import tqdm
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from losses import *
import numpy as np
import torch
from Diffusion import *

from torchlevy import LevyStable
levy = LevyStable()



## Sample visualization.

def visualization(samples, sample_batch_size=64):
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()


def gamma_func(x):
    return torch.tensor(gamma(x))


def get_discrete_time(t, N=1000):
    return N * t


def ddim_score_update2(score_model, sde, x_s, s, t, h=0.6, clamp = 10, device='cuda'):
    score_s = score_model(x_s, s)
    time_step = s-t
    beta_step = sde.beta(s)*time_step
    x_coeff = 1 + beta_step/sde.alpha
    if sde.alpha==2:
        score_coeff2 = torch.pow(beta_step, 2 / sde.alpha) * gamma_func(sde.alpha + 1)

    else:
        score_coeff2 = torch.pow(beta_step, 2/sde.alpha)*torch.pow(time_step, 1-2/sde.alpha)*np.sin(torch.pi/2*(2-sde.alpha))/(2-sde.alpha)*2/torch.pi*gamma_func(sde.alpha+1)
    noise_coeff = torch.pow(beta_step, 1 / sde.alpha)

    e_L = torch.clamp(levy.sample(sde.alpha, 0, size=x_s.shape ).to(device),-clamp,clamp)

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff2[:, None, None, None] * score_s + noise_coeff[:, None, None,None] * e_L
    #print('score_coee', torch.min(score_coeff), torch.max(score_coeff))
    #print('noise_coeff',torch.min(noise_coeff), torch.max(noise_coeff))
    #print('x_coeff', torch.min(x_coeff), torch.max(x_coeff))
    print('x_s range', torch.min(x_s), torch.max(x_s))
    print('x_t range', torch.min(x_t), torch.max(x_t))
    print('x_s mean', torch.mean(x_s))
    print('x_t mean', torch.mean(x_t))
    print('score range',torch.min(score_s), torch.max(score_s))
    #print('x coeff adding', torch.min(x_coeff[:, None, None, None] * x_s), torch.max(x_coeff[:, None, None, None] * x_s))
    #print('score adding',torch.min(score_coeff[:, None, None, None] * score_s), torch.max(score_coeff[:, None, None, None] * score_s) )
    #print('noise adding', torch.min(noise_coeff[:, None, None,None] * e_L), torch.max(noise_coeff[:, None, None,None] * e_L))


    return x_t


def pc_sampler2(score_model,
                sde,
                alpha,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp = 10, datasets="MNIST", clamp_mode = 'constant'):
    t = torch.ones(batch_size, device=device)
    if datasets =="MNIST":
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28,28)).to(device)
        x_s = torch.clamp(e_L,-1,1) *sde.marginal_std(t)[:,None,None,None]
    elif datasets =="CIFAR10":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32,32)).to(device)
        x_s = torch.clamp(e_L,-1.5,1.5) *sde.marginal_std(t)[:,None,None,None]

    elif datasets =="CelebA":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32,32)).to(device)
        x_s = torch.clamp(e_L,-5,5) *sde.marginal_std(t)[:,None,None,None]

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.linspace(1., eps, num_steps)  # (t_{N-1}, t_{N-2}, .... t_0)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)
    #visualization(x_s)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0])*t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0]*(clamp-1)+1

            if Predictor:
                x_s = ddim_score_update2(score_model, sde, x_s, batch_time_step_s, batch_time_step_t, clamp = linear_clamp)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t
    x_t = x_s
    #visualization(x_t, batch_size)
    if trajectory:
        return samples
    else:
        return x_t

def dpm_score_update(model, sde, x_s, s, t, alpha, h=0.06, return_noise=False):
    log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
    lambda_s = sde.marginal_lambda(s)
    lambda_t = sde.marginal_lambda(t)
    sigma_s = sde.marginal_std(s)
    sigma_t = sde.marginal_std(t)
    time_step = s-t
    beta_step = sde.beta(s)*time_step

    score_s = model(x_s, s)

    h_t = lambda_t - lambda_s

    x_coeff = torch.exp(log_alpha_t - log_alpha_s)

    score_coeff = sigma_t * torch.pow(sigma_s, alpha - 1) * alpha * torch.expm1(h_t) \
                  * gamma_func(alpha - 1) / torch.pow(gamma_func(alpha / 2), 2) / np.power(h, alpha - 2)
    score_coeff2 = torch.exp(log_alpha_s - log_alpha_s)* gamma_func(sde.alpha-1)/torch.pow(gamma_func(sde.alpha/2),2)*1/np.power(h ,sde.alpha-2)*beta_step
    x_t = x_coeff[:, None, None, None] * x_s - score_coeff2[:, None, None, None] * score_s

    return x_t

def dpm_sampler(score_model,
                sde,
                alpha,
                batch_size,
                num_steps,
                device='cuda',
                eps=1e-3):
    e_L=levy.sample(sde.alpha, 0, (batch_size, 1, 28,28)).to(device)
    time_steps = np.linspace(1., eps, num_steps)  # (t_{N-1}, t_{N-2}, .... t_0)
    x_s = e_L

    visualization(x_s)
    alpha = sde.alpha
    batch_time_step_s = torch.ones(batch_size, device=device) * time_steps[0]

    with torch.no_grad():
     for time_step in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(batch_size, device=device) * time_step

            x_t = dpm_score_update(score_model, sde, x_s, batch_time_step_s, batch_time_step_t, alpha)

            batch_time_step_s = batch_time_step_t

    return x_t

