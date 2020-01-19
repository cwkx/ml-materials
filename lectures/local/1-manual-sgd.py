# main imports
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# local version imports
import visdom
vis = visdom.Visdom(port=12345)
vis.line(X=np.array([0]), Y=np.array([[np.nan, np.nan]]), win='loss')
vis.line(X=np.array([0]), Y=np.array([[np.nan, np.nan]]), win='acc')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])),
batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.l1 = nn.Linear(in_features=1024, out_features=512)
        self.l2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        return x

N = SimpleNetwork().to(device)

# initialise the optimiser
epoch = 0

# train
while (epoch<100):
    
    # arrays for metrics
    train_loss_arr = np.zeros(0)
    train_acc_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    # iterate over some of train dateset
    for i in range(1000):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        for p in N.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

        p = N(x.view(x.size(0), -1))
        pred = p.argmax(dim=1, keepdim=True)
        loss = torch.nn.functional.cross_entropy(p, t)
        loss.backward()
        
        for p in N.parameters():
            p.data = p.data-0.01*p.grad.data

        train_loss_arr = np.append(train_loss_arr, loss.data)
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    # iterate entire test dataset
    for x,t in test_loader:
        x,t = x.to(device), t.to(device)

        p = N(x.view(x.size(0), -1))
        loss = torch.nn.functional.cross_entropy(p, t)
        pred = p.argmax(dim=1, keepdim=True)

        test_loss_arr = np.append(test_loss_arr, loss.data)
        test_acc_arr = np.append(test_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    # plot metrics
    vis.line(X=np.array([epoch]), Y=np.array([[
        train_loss_arr.mean(),
        test_loss_arr.mean()
    ]]), win='loss', opts=dict(title='loss',xlabel='epoch', ylabel='loss', ytype='log', legend=[
        'train loss',
        'test loss'
    ]), update='append')

    vis.line(X=np.array([epoch]), Y=np.array([[
        train_acc_arr.mean(),
        test_acc_arr.mean()
    ]]), win='acc', opts=dict(title='acc',xlabel='epoch', ylabel='loss', ytype='log', legend=[
        'train accuracy',
        'test accuracy'
    ]), update='append')

    epoch = epoch+1
