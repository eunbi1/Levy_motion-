import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

from sampling import *
from training import *
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


print(device)

#train(beta_min=0.1, alpha=1.9,beta_max=20,n_epochs=1000)
#train(beta_min=1, alpha=1.9, abeta_max=20,n_epochs=1000)
samples = sample(alpha=1.9, path ='/home/eunbiyoon/Levy_motion-/CIFAR101.9_0.1_10.0.pth',
            beta_min=0.1, beta_max=10, sampler='pc_sampler2', batch_size=64, num_steps=1000,LM_steps=20,
            Predictor=True, Corrector=False, trajectory = False, clamp =1,datasets="CIFAR10")
#sample(alpha=1.9, path ="/home/eunbiyoon/Levy_motion-/MNIST1.9_0.1_20.pth",beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64,num_steps=1000,LM_steps=20,Predictor=True, Corrector=False)

