# setup
pip install visdom
source venv/bin/activate
python -m visdom.server -port 12345

# examples
import torch
import visdom
vis = visdom.Visdom(port=12345)

vis.text('hello world')
vis.image(torch.rand(3,256,256))
vis.image(torch.rand(3,256,256))

for i in range(20):
    vis.image(torch.rand(3,256,256), win='blah')

vis.line(torch.rand(20))
vis.histogram(torch.randn(1000))
vis.scatter(torch.randn(10,3))