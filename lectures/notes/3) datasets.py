import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import visdom
vis = visdom.Visdom(port=12345)

# image folder example
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('hymenoptera_data/train', train_transform)
test_dataset = datasets.ImageFolder('hymenoptera_data/val', test_transform)

class_names = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=4, shuffle=True, num_workers=4)

train_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size=4, shuffle=False, num_workers=4)

x,label = next(iter(train_loader))
x.size()

grid = torchvision.utils.make_grid(x)
grid = (grid-grid.min())/(grid.max()-grid.min())
vis.image(grid)

# fashion mnist example
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])),
shuffle=True, batch_size=64, drop_last=True)

grid = torchvision.utils.make_grid(x)
grid = (grid-grid.min())/(grid.max()-grid.min())
vis.image(grid)

# handling text examples on PyTorch
#  - tutorials
#    - text