import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.conv_block(x))

N = nn.Sequential(
    nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    ResidualBlock(128),
    ResidualBlock(128),
    ResidualBlock(128),
    nn.Conv2d(128, 100, 4, stride=2, padding=1, bias=False),
    nn.AvgPool2d(4),
    nn.Flatten(),
).to(device)
