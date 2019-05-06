class ResidualBlock(nn.Module):
   def __init__(self, in_features):
       super(ResidualBlock, self).__init__()

       conv_block = [ nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(in_features) ]

       self.conv_block = nn.Sequential(*conv_block)

   def forward(self, x):
       return torch.relu(x + self.conv_block(x))


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
        layers.append(ResidualBlock(128))
        layers.append(ResidualBlock(128))
        layers.append(ResidualBlock(128))
        layers.append(nn.Conv2d(128, 100, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.AvgPool2d((4,4)))
        self.layers = layers

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

N = ConvolutionalNetwork().to(device)