import torch
from torch import nn
from torch.nn import functional as F
from wgan.generator_critic import Generator, Critic
from wgan.utils import extract_gradients, gradient_penalty, APA

class WGAN(nn.Module):
    def __init__(
        self,
        wgan_type,
        num_classes,
        generator_backbone,
        critic_backbone,
        device,
        aug = lambda x: x
    ):
        super().__init__()

        if wgan_type not in ['ac', 'pc']:
            raise ValueError("WGAN type can only be set to 'ac' (auxiliary classifier) or 'pc' (projection critic)")
        
        self.generator = Generator(backbone = generator_backbone).to(device)
        self.critic = Critic(
            backbone = critic_backbone,
            num_classes = num_classes,
            wgan_type = wgan_type
        ).to(device)
        
        self.device = device
        self.num_classes = num_classes
        self.wgan_type = wgan_type
        if wgan_type == 'ac':
            self.cross_entropy = nn.CrossEntropyLoss()
        
        self.aug = aug
        if isinstance(self.aug, APA):
            self.aug.generate = self.generate

    def generate(self, batch_size = 1, latent = None, classes = None, return_latent = False, return_classes = False):
        if latent == None:
            latent = torch.randn(batch_size, *(self.generator.latent_size)).to(self.device)
        if classes == None:
            classes = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        if return_latent:
            if return_classes:
                return self.generator(latent, classes), latent, classes
            else:
                return self.generator(latent, classes), latent
        else:
            if return_classes:
                return self.generator(latent, classes), classes
            else:
                return self.generator(latent, classes)

    def calc_disc_loss(self, x_real, c_real):
        if isinstance(self.aug, APA):
            x_real = self.aug(x_real, c_real)
        else:
            x_real = self.aug(x_real)
        epsilon = torch.rand((x_real.shape[0], 1, 1, 1), requires_grad = True).to(x_real.device)
        x_fake = self.generate(x_real.shape[0], classes = c_real).detach()
        if not isinstance(self.aug, APA):
            x_fake = self.aug(x_fake)
        gradients = extract_gradients(self.critic, x_real, x_fake, c_real, epsilon)
        scores_fake, class_scores_fake = self.critic(x_fake, c_real)
        scores_real, class_scores_real = self.critic(x_real, c_real)
        fake_loss = scores_fake.mean()
        real_loss = scores_real.mean()
        if self.wgan_type == 'pc':
            clf_loss = class_scores_fake.mean() - class_scores_real.mean()
        elif self.wgan_type == 'ac':
            clf_loss = 0.5 * self.cross_entropy(class_scores_fake, c_real) + 0.5 * self.cross_entropy(class_scores_real, c_real)
        gp = gradient_penalty(gradients)
        return fake_loss - real_loss, clf_loss, gp

    def calc_gen_loss(self, batch_size = 8):
        x_fake, c_fake = self.generate(batch_size, return_classes = True)
        if not isinstance(self.aug, APA):
            x_fake = self.aug(x_fake)
        scores_fake, class_scores_fake = self.critic(x_fake, c_fake)
        fake_loss = -1.0 * scores_fake.mean()
        if self.wgan_type == 'pc':
            clf_loss = -1.0 * class_scores_fake.mean()
        elif self.wgan_type == 'ac':
            clf_loss = self.cross_entropy(class_scores_fake, c_fake)
        return fake_loss, clf_loss
