# traditional gradient clipping, as in "Wasserstein GAN"
def grad_clip(M, clamp_lower=-0.01, clamp_upper=0.01):
    for p in M.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)

# improved Wasserstein gradient penalty (slow) as in "Improved Training of Wasserstein GANs"
def grad_penalty(M, real_data, fake_data, lmbda=10):

    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    lerp = alpha * real_data + ((1 - alpha) * fake_data)
    lerp = lerp.to(device)
    lerp.requires_grad = True
    lerp_d = M.discriminate(lerp)

    gradients = torch.autograd.grad(outputs=lerp_d, inputs=lerp, grad_outputs=torch.ones(lerp_d.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
