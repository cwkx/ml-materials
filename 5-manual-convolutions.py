import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

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

kernel = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]]).reshape(1, 1, 3, 3)
with torch.no_grad():
    conv.weight.copy_(kernel)

grid = torchvision.utils.make_grid(normalise(x))
plt.imsave('convolution-input.png', grid.permute(1, 2, 0).numpy())

grid = torchvision.utils.make_grid(normalise(conv(x)))
plt.imsave('convolution-output.png', grid.detach().permute(1, 2, 0).numpy())
