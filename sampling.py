import os


from model import *
from cifar10_model import *
from Diffusion import *
from sampler import *
import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt

image_size = 28
channels = 1
batch_size = 128

def diffusion_animation(samples, name="diffusion_1.8.gif"):
    fig =plt.figure(figsize=(12, 12))
    batch_size = 64
    ims = []
    for i in range(0,len(samples),10):
     sample = samples[i]
     sample_grid = make_grid(sample, nrow=int(np.sqrt(batch_size)))
     plt.axis('off')
     im=plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
     ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(name)
    plt.show()

def sample(path='ckpt.pth', alpha=2,beta_min=0.1, beta_max=20,
           num_steps = 1000, batch_size = 64, LM_steps=1000, sampler ='pc_sampler2',
           Predictor=True, Corrector=False, name='image', trajectory=False,
           clamp =10, device='cuda', datasets = "MNIST", clamp_mode="constant"):

    sde = VPSDE(alpha=alpha,beta_min=beta_min, beta_max=beta_max)


    if datasets == "CIFAR10":
        score_model = Model()
        
    if datasets == "CelebA":
        score_model = Model()

    if datasets == "MNIST":
        score_model = Unet(
            dim=28,
            channels=1,
            dim_mults=(1, 2, 4,))

    score_model = score_model.to(device)
    ckpt = torch.load(path, map_location=device)
    score_model.load_state_dict(ckpt,  strict= False)
    score_model.eval()

    if sampler =='pc_sampler2':
        samples = pc_sampler2(score_model,
                          sde=sde, alpha=sde.alpha,
                          batch_size=batch_size,
                          num_steps=num_steps,
                          device=device,
                          Predictor=Predictor, Corrector=Corrector,
                          LM_steps = LM_steps, trajectory=trajectory,
                              clamp = clamp, datasets = datasets, clamp_mode=clamp_mode)
    elif sampler == 'dpm_sampler':
        samples = dpm_sampler(score_model,
                          sde=sde, alpha=sde.alpha,
                          batch_size=batch_size,
                          num_steps=num_steps,
                          device=device)
    if trajectory:
        for i, img in enumerate(samples):
            img = (img+1)/2
            img = img.clamp(0.0, 1.0)
            samples[i]= img
        last_sample = samples[-1]
    else:
        samples = (samples+1)/2
        samples = samples.clamp(0.0, 1.0)
        last_sample = samples

    sample_grid = make_grid(last_sample, nrow=int(np.sqrt(batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

    name = str(datasets)+str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_'+ 'alpha'+str(f'{alpha}')+'beta'+ str(f'{beta_min}') + '_'+str(
        f'{beta_max}') + '.png'

    dir_path = os.path.join('/scratch/private/eunbiyoon/Levy_motion-', 'sample')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    name = os.path.join(dir_path, name)
    plt.savefig(name)
    plt.show()

    if trajectory:
        name2 = str(datasets)+ str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_' + 'alpha' + str(
            f'{alpha}') + 'beta' + str(f'{beta_min}') + '_' + str(
            f'{beta_max}') + '.gif'
        name2 = os.path.join(dir_path,name2)
        diffusion_animation(samples, name=name2)

    return samples
