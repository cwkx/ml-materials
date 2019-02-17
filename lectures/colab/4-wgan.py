%%capture
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'
!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
!pip install livelossplot

# main imports
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import sleep
from livelossplot import PlotLosses

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# import dataset
# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=False, batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# view some of the dataset
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_loader.dataset[i][0].permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.xlabel(class_names[test_loader.dataset[i][1]])

# improved wasserstein gradient penalty
def grad_penalty(M, real_data, fake_data, lmbda=10):

    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    lerp = alpha * real_data + ((1 - alpha) * fake_data)
    lerp = lerp.to(device)
    lerp.requires_grad = True
    lerp_d = M.discriminate(lerp)

    gradients = torch.autograd.grad(outputs=lerp_d, inputs=lerp, grad_outputs=torch.ones(lerp_d.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda

# define two models: (1) Generator, and (2) Discriminator
# define the model
class Generator(nn.Module):
    def __init__(self, f=64):
        super(Generator, self).__init__()
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(100, f*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(f*8, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(f*4, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(f*2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(True),
            nn.ConvTranspose2d(f, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

class Discriminator(nn.Module):
    def __init__(self, f=64):
        super(Discriminator, self).__init__()
        self.discriminate = nn.Sequential(
            nn.Conv2d(3, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f*2, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f*4, f*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f*8, 1, 4, 2, 1, bias=False)
            # nn.Sigmoid() # NOPE! for wasserstein
        )
        
G = Generator().to(device)
D = Discriminator().to(device)

print(f'> Number of generator parameters {len(torch.nn.utils.parameters_to_vector(G.parameters()))}')
print(f'> Number of discriminator parameters {len(torch.nn.utils.parameters_to_vector(D.parameters()))}')

# initialise the optimiser
optimiser_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
optimiser_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.9))
bce_loss = nn.BCELoss()
epoch = 0
liveplot = PlotLosses()

# main training loop
# training loop
while (epoch<50):
    
    # arrays for metrics
    logs = {}
    gen_loss_arr = np.zeros(0)
    dis_loss_arr = np.zeros(0)
    grad_pen_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(200):

        # train discriminator k times
        for k in range(5):
          
            x,t = next(train_iterator)
            x,t = x.to(device), t.to(device)
            optimiser_D.zero_grad()

            g = G.generate(torch.randn(x.size(0), 100, 1, 1).to(device))
            l_r = D.discriminate(x).mean()
            l_r.backward(-1.0*torch.ones(1)[0].to(device)) # real -> -1
            l_f = D.discriminate(g.detach()).mean()
            l_f.backward(torch.ones(1)[0].to(device)) #  fake -> 1
                        
            loss_d = (l_f - l_r)
            grad_pen = grad_penalty(D, x.data, g.data, lmbda=10)
            grad_pen.backward()
            
            optimiser_D.step()
            
            dis_loss_arr = np.append(dis_loss_arr, loss_d.cpu().data)
            grad_pen_arr = np.append(grad_pen_arr, grad_pen.cpu().data)
        
        # train generator
        optimiser_G.zero_grad()
        g = G.generate(torch.randn(x.size(0), 100, 1, 1).to(device))
        loss_g = D.discriminate(g).mean()
        loss_g.backward(-1.0*torch.ones(1)[0].to(device)) # fake -> -1
        optimiser_G.step()
        
        gen_loss_arr = np.append(gen_loss_arr, -loss_g.cpu().data)

    # plot some examples
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)

    liveplot.update({
        'generator loss': gen_loss_arr.mean(),
        'discriminator loss': dis_loss_arr.mean(),
        'grad penalty': grad_pen_arr.mean(),
    })
    liveplot.draw()
    sleep(1.)

    epoch = epoch+1