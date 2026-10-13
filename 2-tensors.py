import torch

x = torch.zeros(10)
x
x.size()

# basic operations and broadcasting
x+2
x+3 * torch.eye(10)

x = torch.rand(10,5)
x
x.size()

# adding/removing dimensions
x.unsqueeze(0).size()
x.unsqueeze(1).size()
x.unsqueeze(2).size()

x.unsqueeze_(1)
x.size()

x = x.unsqueeze(3)
x.size()

x.squeeze(3).size()
x = x.repeat(1,3,1,4)

# indexing
x[0].size()
x[:,0].size()
x[:3].size()
x[3:].size()

# viewing
x.view(-1).size()
x = x.view(x.size(0), -1)
x.size()

x = x.view(10, 60, 1, 1)
x.size()
x = x.squeeze()
x.size()

# types
x = (x*10).long()
x
x.dtype
x = x.float()
x.dtype

# gpu
x.device
x = x.cuda()
x.device
x+2
x.device

# compilation
@torch.compile
def process(x):
    a = x * 2
    b = a + 3
    return torch.relu(b)

x = torch.randn(1024, device='cpu')
y = process(x)  # Warmup (trace, compile, cache, execute)
y = process(x)  # Cached reuse of compiled code
print('Compiled output:', y.shape, y.device)
torch.allclose(y, torch.relu(x * 2 + 3))
