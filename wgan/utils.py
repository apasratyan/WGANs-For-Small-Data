import torch
from torch import nn

def extract_gradients(critic, x_real, x_fake, c_real, epsilon):
    mixed_images = x_real * epsilon + x_fake * (1 - epsilon)
    mixed_images = torch.autograd.Variable(mixed_images, requires_grad = True)

    mixed_scores, mixed_class_scores = critic(mixed_images, c_real)
    if critic.wgan_type == 'pc':
        mixed_scores = torch.concatenate([mixed_scores, mixed_class_scores], dim = 0)
    
    gradients = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    return gradients

def gradient_penalty(gradients):
    gradients = gradients.view(len(gradients), -1)

    gradient_norm = torch.sqrt((gradients ** 2).sum(dim = 1) + 1e-12)

    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty

class APA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.generate = None
        self.p = torch.tensor(0.0)

    def forward(self, x, c):
        ids = (torch.rand(x.shape[0]) <= self.p)
        batch_size = ids.int().sum()
        with torch.no_grad():
            gen = self.generate(batch_size = batch_size, classes = c[ids])
        x[ids] = gen
        return x

    def set_p(self, new_p):
        self.p.data = torch.tensor(new_p)
