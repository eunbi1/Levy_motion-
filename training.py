import os
from sampling import *
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10,CelebA
import tqdm
import os
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import time
from model import *
from losses import *
from Diffusion import *
import numpy as np
import torch
from cifar10_model import *
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize





torch.multiprocessing.set_start_method('spawn')
from torchlevy import LevyStable

levy = LevyStable()

image_size = 28
channels = 1
batch_size = 128


def train(alpha=2, lr = 1e-4, batch_size=128, beta_min=0.1, beta_max = 20,
          n_epochs=15,num_steps=1000, datasets ='MNIST',path = None, device='cuda'):
    sde = VPSDE(alpha, beta_min = beta_min, beta_max = beta_max, device=device)

    if device == 'cuda':
        num_workers =0
    else:
        num_workers = 4

    if datasets =="MNIST":
        dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,generator=torch.Generator(device=device))
        image_size = 28
        channels = 1
        score_model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,))

    if datasets == "CelebA":
        image_size = 32
        channels = 3

        transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),])
        dataset = CelebA('/scratch/private/eunbiyoon/Levy_motion-', transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,generator=torch.Generator(device=device))

        score_model = Model(resolution=image_size, in_channels=channels,out_ch=channels)

    if datasets =='CIFAR10':
        dataset = CIFAR10('.', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,generator=torch.Generator(device=device))
        image_size = 32
        channels = 3
        score_model = Model()


    score_model = score_model.to(device)
    if path:
      ckpt = torch.load(path, map_location=device)
      score_model.load_state_dict(ckpt,  strict=False)

    optimizer = Adam(score_model.parameters(), lr=lr)


    L = []
    counter = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        counter += 1
        avg_loss = 0.
        num_items = 0
        i=0
        for x,y in data_loader:
            x = 2*x - 1
            x = x.to(device)
            e_L = torch.clamp(levy.sample(alpha, 0, size=x.shape ).to(device),-10,10)

            t = torch.rand(x.shape[0]).to(device)+1e-4
            loss = loss_fn(score_model, sde, x, t, e_L, num_steps=num_steps)

            optimizer.zero_grad()
            loss.backward()
            #print(f'{epoch} th epoch {i} th step loss: {loss}')
            i +=1
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        t1 = time.time()
        L.append(avg_loss / num_items)
        print(f'{epoch} th epoch loss: {avg_loss / num_items}')
        print('Running time:', t1-t0)
        ckpt_name = str(datasets)+ str(f'{alpha}_{beta_min}_{beta_max}.pth')
        ckpt_name=os.path.join('/scratch/private/eunbiyoon/Levy_motion-', ckpt_name)
        torch.save(score_model.state_dict(),ckpt_name)
    name = str(datasets)+str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) +'_'+ 'alpha'+str(f'{alpha}')+ 'beta'+str(f'{beta_min}')+ '_'+ str(
            f'{beta_max}') + '.pth'
    dir_path = os.path.join('/scratch/private/eunbiyoon/Levy_motion-', 'checkpoint')

    if not os.path.isdir(dir_path):
            os.mkdir
