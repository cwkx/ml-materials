import torch

x = torch.zeros(16,16)

x.requires_grad # defaults to False

z = x+2

# without setting requires_grad, the tensor simply acts as a multi-dimensional array with nothing special
z
z.grad_fn # empty
x.grad # empty

# however, setting requires_grad to True...
del x
x = torch.zeros(16,16)
x.requires_grad = True

z = x+2
z.grad_fn # <AddBackward at 0x7f64cafd4a58>

z = z*3
z.grad_fn # <MulBackward at 0x7f64cb5879b0>
z.grad_fn.next_functions[0][0] # <AddBackward at 0x7f64cafd4a58>

out = z.mean()
out # tensor(6., grad_fn=<MeanBackward1>)

x.grad # empty

out.backward()

x.grad 