from torch import nn
from torch.nn import functional as F
from wgan.resblock import CondBN, GeneratorResBlock, CriticResBlock
import numpy as np

class Generator(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.latent_size = backbone.in_shape

        self.backbone = backbone
        
        self.act = nn.ReLU()
        self.norm = CondBN(backbone.out_shape[0], 2)
        self.final_conv = nn.Conv2d(in_channels = backbone.out_shape[0], out_channels = 3, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, z, c):
        x = self.backbone(z, c)
        x = self.norm(self.act(x), c)
        x = self.final_conv(x)
        return F.tanh(x)

class Critic(nn.Module):
    def __init__(self, backbone, num_classes, wgan_type):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.wgan_type = wgan_type

        self.scorer = nn.Linear(np.prod(backbone.out_shape), 1, bias = False)
        if wgan_type == 'pc':
            self.class_embs = nn.Embedding(num_classes, np.prod(backbone.out_shape))
            self.class_embs.weight.data.fill_(0)
        elif self.wgan_type == 'ac':
            self.classifier = nn.Linear(np.prod(backbone.out_shape), num_classes)
            self.classifier.weight.data.fill_(0)
            self.classifier.bias.data.fill_(0)

    def forward(self, x, c):
        x = self.backbone(x).view(x.shape[0], -1)

        scores = self.scorer(x)
        if self.wgan_type == 'pc':
            class_scores = (self.class_embs(c) * x).sum(dim = 1, keepdim = True)
        elif self.wgan_type == 'ac':
            self.class_scores = nn.Linear(x)
        return scores, class_scores
