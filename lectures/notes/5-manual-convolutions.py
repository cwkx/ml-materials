import visdom
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

vis = visdom.Visdom(port=12345)

def normalise(x):
    return (x-x.min())/(x.max()-x.min())

# example data for this tutorial
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=64, drop_last=True)
x, label = next(iter(train_loader))

x.size() # 64, 1, 32, 32

# tutorial start
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

conv.weight.size()

conv.weight.data = torch.tensor([[-1,0.,1],
                                 [-2,0.,2],
                                 [-1,0.,1]]).unsqueeze(0).unsqueeze(0)

vis.image(torchvision.utils.make_grid(normalise(x)))
vis.image(torchvision.utils.make_grid(normalise(conv(x))))