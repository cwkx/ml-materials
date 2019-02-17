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
from livelossplot import PlotLosses

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
shuffle=True, batch_size=16, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=False, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# define unet model
class UNet(nn.Module):
  
    def __init__(self, in_class=1, out_class=3):
        super().__init__()
                
        self.dconv_down1 = self.conv_block(in_class, 64)
        self.dconv_down2 = self.conv_block(64, 128)
        self.dconv_down3 = self.conv_block(128, 256)
        self.dconv_down4 = self.conv_block(256, 512)
        
        self.dconv_up3 = self.conv_block(256 + 512, 256)
        self.dconv_up2 = self.conv_block(128 + 256, 128)
        self.dconv_up1 = self.conv_block(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, out_class, 1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    )
        
    def forward(self, x):
        conv1 = self.dconv_down1(x) # skip-connection 1
        x = F.max_pool2d(conv1, 2)
        conv2 = self.dconv_down2(x) # skip-connection 2
        x = F.max_pool2d(conv2, 2)
        conv3 = self.dconv_down3(x) # skip-connection 3
        x = F.max_pool2d(conv3, 2)
        
        x = self.dconv_down4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        
        return out

N = UNet().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
epoch = 0
liveplot = PlotLosses()

# we're going to go from grayscale to color. To create a training set, we'll write color to grayscale function
def grayscale(x):
    rgb = (0.299 * x[:,0] + 0.587 * x[:,1] + 0.114 * x[:,2])
    return rgb.unsqueeze(1)

# training loop, feel free to also train on the test dataset if you like
while (epoch<5):
    
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(1000):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        optimiser.zero_grad()
        p = N(grayscale(x)) # color prediction from grayscale input
        loss = ((p-x)**2).mean()
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    # NOTE: live plot library has dumb naming forcing our 'test' to be called 'validation'
    liveplot.update({
        'loss': train_loss_arr.mean()
    })
    liveplot.draw()

    epoch = epoch+1

# input grayscale images
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(grayscale(x)).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)

# model prediction
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(p).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)

# ground truth
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
