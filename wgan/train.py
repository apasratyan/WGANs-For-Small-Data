import torch
from torch import nn, optim
import os, shutil
from PIL import Image
from IPython.display import clear_output
from pytorch_fid.fid_score import calculate_fid_given_paths
from ada import AdaptiveDiscriminatorAugmentation
from wgan.utils import APA
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_model_samples(name, model, batch_size, num_samples):
    if os.path.exists(name):
        shutil.rmtree(name)

    os.makedirs(name, exist_ok=True)
    count = 0

    while count < num_samples:
        cur_batch_size = min(num_samples - count, batch_size)
        classes = (torch.rand(cur_batch_size) <= (162 / 206)).int().to(model.device)
        with torch.no_grad():
            out = model.generate(cur_batch_size, classes = classes)

        out = (out * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i in range(out.shape[0]):
            img = Image.fromarray(out[i])
            n_digits = len(str(count))
            img_name = (6 - n_digits) * '0' + str(count) + '.png'
            img.save(os.path.join(name, img_name))
            count += 1

def train_wgan(
    model,
    optimizers,
    schedulers,
    loader,
    loss_logs,
    fid_logs,
    save_path,
    test_path,
    sample_count = 100,
    aug_f = lambda x: x,
    n_iters = 100000,
    disc_iters = 1,
    show_every = 1000,
    save_every = 1000,
):
    for i in (pbar := tqdm(range(n_iters))):
        model.train()

        avg_disc_loss = 0.0
        avg_clf_disc_loss = 0.0
        avg_gp = 0.0
        for n in range(disc_iters):
            batch, classes = loader.__iter__().__next__()
            batch = batch.to(model.device)
            classes = classes.to(model.device)
            optimizers['crt'].zero_grad()
            optimizers['gen'].zero_grad()
            disc_loss, clf_disc_loss, gp = model.calc_disc_loss(batch, classes)
            loss = 1.0 * disc_loss + clf_disc_loss + 10.0 * gp
            loss.backward()
            optimizers['crt'].step()

            avg_disc_loss += disc_loss.item() / disc_iters
            avg_clf_disc_loss += clf_disc_loss.item() / disc_iters
            avg_gp += gp.item() / disc_iters
        
        optimizers['crt'].zero_grad()
        optimizers['gen'].zero_grad()
        gen_loss, clf_gen_loss = model.calc_gen_loss(loader.batch_size)
        loss = 1.0 * gen_loss + 1.0 * clf_gen_loss
        loss.backward()
        optimizers['gen'].step()

        loss_logs[0].append(avg_disc_loss)
        loss_logs[1].append(avg_clf_disc_loss)
        loss_logs[2].append(clf_gen_loss.item())
        loss_logs[3].append(avg_gp)
        loss_logs[4].append(model.aug.p.item())
        
        if isinstance(model.aug, AdaptiveDiscriminatorAugmentation) or isinstance(model.aug, APA):
            if i >= 5:
                adj = np.sign(aug_f(np.mean(loss_logs[3][-5:])) - model.aug.p.item())
                new_p = np.clip(model.aug.p.item() + adj * (disc_iters * 16) / (100 * 1000), 0.1, 1)
                model.aug.set_p(new_p)

        model.eval()
        
        pbar.set_description(f"disc/clf/p: {disc_loss.item():.5f}/{clf_disc_loss.item():.5f}/{model.aug.p.item():5f}")
        if (i + 1) % show_every == 0:
            with torch.no_grad():
                generated = model.generate(batch_size = 20, classes = torch.arange(2).repeat(10).to(model.device)).cpu().detach()
            clear_output()
            fig, axs = plt.subplots(1, 2, figsize = (20, 10))
            axs[0].grid()
            axs[0].plot(np.convolve(loss_logs[0], np.ones(100) / 100, mode = 'valid'), label = 'disc loss', linewidth = 0.5)
            axs[0].plot(np.convolve(loss_logs[1], np.ones(100) / 100, mode = 'valid'), label = 'clf disc loss', linewidth = 0.5)
            axs[0].plot(np.convolve(loss_logs[3], np.ones(100) / 100, mode = 'valid'), label = 'gradient penalty', linewidth = 0.5)
            axs[0].legend()
            axs[1].imshow(make_grid(generated * 0.5 + 0.5, nrow = 2).permute(1, 2, 0))
            axs[1].axis('off')
            plt.tight_layout()
            plt.show()

        if (i + 1) % save_every == 0:
            os.makedirs(save_path + '/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), save_path + '/checkpoints/weights.ckpt')

            save_model_samples(save_path + '/samples', model, 64, sample_count)
            fid = calculate_fid_given_paths([save_path + '/samples', test_path], 64, model.device, 2048)
            fid_logs.append(fid)

        schedulers['crt'].step()
        schedulers['gen'].step()
