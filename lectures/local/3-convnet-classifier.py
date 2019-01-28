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

class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])),
shuffle=True, batch_size=64, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])),
batch_size=64, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# define the model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 100, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.AvgPool2d((4,4)))
        self.layers = layers

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

N = ConvolutionalNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.0001)
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

        optimiser.zero_grad()
        p = N(x).view(x.size(0), len(class_names))
        pred = p.argmax(dim=1, keepdim=True)
        loss = torch.nn.functional.cross_entropy(p, t)
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.data)
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    # iterate entire test dataset
    for x,t in test_loader:
        x,t = x.to(device), t.to(device)

        p = N(x).view(x.size(0), len(class_names))
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