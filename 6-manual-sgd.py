import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('teaching.mplstyle')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print('Training examples:', len(train_dataset))
print('Test examples:', len(test_dataset))

def cycle(loader):
    while True:
        yield from loader

train_iterator = cycle(train_loader)

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32 * 32, 512)
        self.l2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.l1(x))
        return self.l2(x)

N = SimpleNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

# train
for step in range(1, 101):
    N.train()
    train_loss = train_correct = train_count = 0
    for _ in range(100):
        x, t = next(train_iterator)
        x, t = x.to(device), t.to(device)

        N.zero_grad(set_to_none=True)
        logits = N(x)
        loss = loss_fn(logits, t)
        loss.backward()

        with torch.no_grad():
            for parameter in N.parameters():
                parameter.add_(parameter.grad, alpha=-0.01)

        train_loss += loss.item() * len(t)
        train_correct += (logits.argmax(dim=1) == t).sum().item()
        train_count += len(t)

    # evaluate
    N.eval()
    test_loss = test_correct = test_count = 0
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            logits = N(x)
            loss = loss_fn(logits, t)
            test_loss += loss.item() * len(t)
            test_correct += (logits.argmax(dim=1) == t).sum().item()
            test_count += len(t)

    train_losses.append(train_loss / train_count)
    test_losses.append(test_loss / test_count)
    train_accuracies.append(train_correct / train_count)
    test_accuracies.append(test_correct / test_count)
    steps = range(1, step + 1)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, train_losses, '.-', label='Train')
    axes[0].plot(steps, test_losses, '.-', label='Test')
    axes[0].set(title='Loss', xlabel='Update (100 batches)', ylabel='Cross-entropy', yscale='log')
    axes[0].legend()

    axes[1].plot(steps, train_accuracies, '.-', label='Train')
    axes[1].plot(steps, test_accuracies, '.-', label='Test')
    axes[1].set(title='Accuracy', xlabel='Update (100 batches)', ylabel='Accuracy', ylim=(0, 1))
    axes[1].legend()

    fig.savefig('manual-sgd-metrics.png')
    plt.close(fig)
