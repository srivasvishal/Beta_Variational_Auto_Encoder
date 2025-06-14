#!/usr/bin/env python3
import os
import argparse
import random
import re

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision.datasets as tvd
import torchvision.transforms as T

from vae_net import VAE
from metrics import compute_beta_vae_score, compute_mig
from utils.test import test_function


class DSpritesDataset(Dataset):
    """Loader for dSprites .npz (imgs + latent factors)."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True, encoding='latin1')
        self.imgs = data['imgs'].astype(np.float32)  # (737280,64,64)
        self.latents_classes = data['latents_classes']  # (737280,6)
        meta = data['metadata'].item()
        self.latents_sizes = meta['latents_sizes']  # [1,3,6,40,32,32]
        # Map every full 6‐tuple → one index
        self._factor_to_index = {
            tuple(int(x) for x in vec): i
            for i, vec in enumerate(self.latents_classes)
        }
        self.N = self.imgs.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx]).unsqueeze(0)  # (1,64,64)

    def sample_factors(self, num_samples: int, random_state=0):
        rng = np.random.RandomState(random_state)
        idxs = rng.choice(self.N, size=num_samples, replace=False)
        return self.latents_classes[idxs]  # (num_samples,6)

    def sample_observations_from_factors(self, factors: np.ndarray, random_state=0):
        idxs = [self._factor_to_index[tuple(int(x) for x in f)] for f in factors]
        imgs = self.imgs[idxs]  # (num_samples,64,64)
        return torch.from_numpy(imgs).unsqueeze(1)  # (num_samples,1,64,64)


def load_binary_mnist(data_dir):
    base = os.path.join(data_dir, 'BinaryMNIST')
    train = np.concatenate([
        np.loadtxt(os.path.join(base, 'binarized_mnist_train.amat')),
        np.loadtxt(os.path.join(base, 'binarized_mnist_valid.amat'))
    ])
    test = np.loadtxt(os.path.join(base, 'binarized_mnist_test.amat'))
    return train, test


def generate_latent_traversals(model, n_z, out_type, filename, n_samples=5, size=64):
    """
    Draws a grid of latent traversals (rows = latent dims, cols = -3..+3)
    and saves it to `filename`.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1) collect reconstructions for each (dim, value)
    imgs = []
    with torch.no_grad():
        z0 = torch.randn(1, n_z, device=device)
        for dim in range(n_z):
            for val in torch.linspace(-3, 3, n_samples, device=device):
                z = z0.clone()
                z[0, dim] = val
                recon, _ = model.decode(z)  # recon: (1, H*W) or (1, H, W)
                # Include channel dimension explicitly
                img = recon.view(1, 1, size, size)  # (1, 1, H, W)
                imgs.append(img)

    # 2) cat into one batch: (n_z*n_samples, 1, H, W)
    batch = torch.cat(imgs, dim=0)

    # 3) build a single canvas: shape (3, H_total, W_total)
    canvas = make_grid(batch, nrow=n_samples, normalize=True, pad_value=1)

    # 4) convert to numpy and take only first channel
    arr = canvas[0].cpu().numpy()  # shape (H_total, W_total)

    # 5) plot & save
    plt.figure(figsize=(n_samples, n_z))
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    model.train()


# Weight initialization function
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=('mnist', 'fashionmnist', 'ff', 'dsprites'),
                        required=True)
    parser.add_argument('--dsprites_path',
                        default='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_dir', default='saved_models')
    parser.add_argument('--decoder_type',
                        choices=('bernoulli', 'gaussian'),
                        default='bernoulli')
    parser.add_argument('--Nz', type=int, default=20)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--capacity', type=int, default=None,
                        help='Epochs to ramp KL weight from 0→β')
    parser.add_argument('--eval_freq', type=int, default=10)
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    scaled_mean = True  # Default for most datasets

    if args.dataset == 'mnist':
        dim, hid = 28 * 28, 500
        train_np, test_np = load_binary_mnist(args.data_dir)
        train_tensor = torch.from_numpy(train_np).float()
        test_set = test_np
        train_loader = DataLoader(train_tensor,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        train_num, test_num = train_np.shape[0], test_np.shape[0]

    elif args.dataset == 'fashionmnist':
        dim, hid = 28 * 28, 500
        scaled_mean = False  # Important for Gaussian decoder
        if args.decoder_type != 'gaussian':
            raise ValueError("Fashion-MNIST requires gaussian decoder")

        # Use normalization for [-1, 1] range
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
        ])

        tr = tvd.FashionMNIST(root=args.data_dir, train=True,
                              download=True, transform=transform)
        te = tvd.FashionMNIST(root=args.data_dir, train=False,
                              download=True, transform=transform)

        # Convert to numpy arrays and flatten
        X_train = tr.data.numpy().astype(np.float32)
        X_test = te.data.numpy().astype(np.float32)
        X_train = X_train.reshape(-1, dim)
        X_test = X_test.reshape(-1, dim)

        train_tensor = torch.from_numpy(X_train).float()
        test_set = X_test
        train_loader = DataLoader(train_tensor,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        train_num, test_num = X_train.shape[0], X_test.shape[0]

    elif args.dataset == 'ff':
        dim, hid = 560, 200
        mat = scipy.io.loadmat(os.path.join(args.data_dir,
                                            'Frey_Face', 'frey_rawface.mat'))
        ff = mat['ff'].T / 256.0
        idxs = list(range(ff.shape[0]))
        test_idxs = random.sample(idxs, 281)
        train_idxs = list(set(idxs) - set(test_idxs))
        train_tensor = torch.from_numpy(ff[train_idxs]).float()
        test_set = ff[test_idxs]
        train_loader = DataLoader(train_tensor,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        train_num, test_num = len(train_idxs), len(test_idxs)

    else:  # dsprites
        dim, hid = 64 * 64, 500
        npz = np.load(args.dsprites_path, allow_pickle=True, encoding='latin1')
        imgs = npz['imgs'].astype(np.float32)  # (737280,64,64)
        flat = imgs.reshape(imgs.shape[0], -1)  # (737280,4096)
        train_tensor = torch.from_numpy(flat).float()
        test_set = flat
        train_loader = DataLoader(train_tensor,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        train_num = test_num = flat.shape[0]

    # ── Device setup ─────────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
        print(f'Using CUDA:{args.device} {torch.cuda.get_device_name(args.device)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple MPS backend')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # ── Model & optimizer ───────────────────────────────────────────────────
    net = VAE(args, d=dim, h_num=hid, scaled=scaled_mean)

    # Apply weight initialization
    net.apply(init_weights)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # Adam with lower LR

    # ── Training & evaluation ───────────────────────────────────────────────
    for epoch in range(args.epochs):
        net.set_epoch(epoch)
        net.train()
        run_loss = run_recon = run_kl = 0.0
        with tqdm(total=train_num, desc=f'Epoch {epoch + 1}/{args.epochs}') as pbar:
            for batch in train_loader:
                x = batch.to(device).float()
                optimizer.zero_grad()
                out = net(x)

                if args.decoder_type == 'gaussian':
                    loss, recon, kl = net.compute_loss(
                        x, None, out[1], out[3], out[0], out[2]
                    )
                else:
                    loss, recon, kl = net.compute_loss(
                        x, out[0], out[1], out[3]
                    )

                # Check for NaN values
                if torch.isnan(loss):
                    print("NaN detected in loss! Skipping batch")
                    continue

                loss.backward()
                optimizer.step()

                run_loss += loss.item()
                run_recon += recon.item()
                run_kl += kl.item()

                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    recon=f"{recon.item():.3f}",
                    kl=f"{kl.item():.3f}"
                )
                pbar.update(x.size(0))

        print(f"[Epoch {epoch + 1}] Loss: {run_loss / len(train_loader):.3f}, "
              f"Recon: {run_recon / len(train_loader):.3f}, "
              f"KL: {run_kl / len(train_loader):.3f}")

        if (epoch + 1) % args.eval_freq == 0:
            net.eval()
            test_elbo = test_function(
                net,
                test_num,
                args.dataset,
                args.decoder_type,
                test_set,
                device
            )
            print(f"Epoch {epoch + 1}: test ELBO = {test_elbo:.3f}")

            if args.dataset == 'dsprites':
                ds = DSpritesDataset(args.dsprites_path)
                b_score = compute_beta_vae_score(ds, net)
                mig_score = compute_mig(ds, net)
                print(f"β-VAE score: {b_score:.3f}, MIG: {mig_score:.3f}")
                os.makedirs('results', exist_ok=True)
                generate_latent_traversals(
                    net, args.Nz, args.decoder_type,
                    f"results/traversals_epoch{epoch + 1}.png"
                )

    # ── Save final checkpoint ────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = f"vae_Nz_{args.Nz}_dataset_{args.dataset}_decoder_{args.decoder_type}.pth"
    torch.save(net.state_dict(), os.path.join(args.save_dir, ckpt))
    print(f"Saved model to {args.save_dir}/{ckpt}")