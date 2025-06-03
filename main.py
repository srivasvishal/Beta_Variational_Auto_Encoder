

import os
import random
import time
import argparse
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from vae_net import VAE
from utils.test import test_function
from metrics import compute_beta_vae_score, compute_mig


class DSpritesDataset(Dataset):

    def __init__(self, npz_path: str):
        # — Load the NPZ with allow_pickle=True so we can read the 'metadata' dict —
        data = np.load(npz_path, allow_pickle=True, encoding='latin1')
        # metadata = data['metadata'].item()


        self.imgs = data['imgs'].astype(np.float32)
        self.latents_classes = data['latents_classes']
        metadata = data['metadata'].item()
        self.latents_sizes = metadata['latents_sizes']
        self._factor_to_index = {}
        for idx, factor_vec in enumerate(self.latents_classes):
            self._factor_to_index[tuple(int(x) for x in factor_vec)] = idx

        self.N = self.imgs.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):

        img = self.imgs[idx]  # shape: (64, 64)
        return torch.from_numpy(img).unsqueeze(0)  # shape: (1, 64, 64)

    def sample_factors(self, num_samples: int, random_state=0):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(self.N, size=num_samples, replace=False)
        return self.latents_classes[indices]

    def sample_observations_from_factors(self, factors: np.ndarray, random_state=0):

        idx_list = []
        for f_vec in factors:
            key = tuple(int(x) for x in f_vec.tolist())
            idx_list.append(self._factor_to_index[key])
        idx_arr = np.array(idx_list, dtype=np.int64)
        imgs = self.imgs[idx_arr]  # shape: (num_samples, 64, 64)
        return torch.from_numpy(imgs).unsqueeze(1)  # shape: (num_samples, 1, 64, 64)


def load_binary_mnist(d_dir):

    train_file = os.path.join(d_dir, 'BinaryMNIST', 'binarized_mnist_train.amat')
    valid_file = os.path.join(d_dir, 'BinaryMNIST', 'binarized_mnist_valid.amat')
    test_file  = os.path.join(d_dir, 'BinaryMNIST', 'binarized_mnist_test.amat')

    mnist_train = np.concatenate([np.loadtxt(train_file), np.loadtxt(valid_file)], axis=0)
    mnist_test  = np.loadtxt(test_file)
    return mnist_train, mnist_test


def generate_latent_traversals(model, n_z, out_type, filename, n_samples=5, size=28):

    model.eval()
    device = next(model.parameters()).device
    z0 = torch.randn(1, n_z, device=device)

    grid_imgs = []
    with torch.no_grad():
        for dim in range(n_z):
            # linearly spaced values from -3 to +3
            values = torch.linspace(-3, 3, n_samples, device=device)
            for val in values:
                z_trav = z0.clone()
                z_trav[0, dim] = val

                if out_type == 'gaussian':
                    mu_x, _ = model.decode(z_trav)
                    recon = mu_x
                else:
                    recon, _ = model.decode(z_trav)
                img = recon.view(1, 1, size, size)
                grid_imgs.append(img)


    grid = torch.cat(grid_imgs, dim=0)

    grid_img = make_grid(grid, nrow=n_samples, normalize=True, pad_value=1.0)

    plt.figure(figsize=(n_samples, n_z))

    plt.imshow(grid_img.permute(1, 2, 0).cpu(), cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Training VAE / β-VAE")
    parser.add_argument("--dataset", default='mnist', dest='dataset',
                        choices=('mnist', 'ff', 'dsprites'),
                        help="Dataset to train the VAE")
    parser.add_argument("--dsprites_path", dest='dsprites_path',
                        default="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
                        help="Path to the dSprites .npz file")
    parser.add_argument("--data_dir", dest='data_dir', default="./data",
                        help="The root directory of your dataset")
    parser.add_argument("--epochs", dest='num_epochs', default=5, type=int,
                        help="Total number of epochs to train")
    parser.add_argument("--batch_size", dest="batch_size", default=100, type=int,
                        help="Batch size for training")
    parser.add_argument('--device', dest='device', default=0, type=int,
                        help='Index of CUDA device (use -1 for CPU/MPS auto)')
    parser.add_argument("--save_dir", dest='save_dir', default="./saved_models",
                        help="Directory to save the final trained model")
    parser.add_argument('--decoder_type', dest='decoder_type', default='bernoulli', type=str,
                        choices=('bernoulli', 'gaussian'),
                        help='Decoder type (Bernoulli or Gaussian)')
    parser.add_argument("--Nz", default=20, type=int,
                        help="Dimensionality of the latent code")
    parser.add_argument('--beta', type=float, default=1.0,
                        help='β parameter for β-VAE (default = 1.0)')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Number of epochs to reach full KL capacity (None for fixed β)')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluate (ELBO + metrics) every eval_freq epochs')
    args = parser.parse_args()

    # ─── Load data & define shapes ─────────────────────────────────────────
    if args.dataset == 'mnist':
        dim = 28 * 28
        hid_num = 500
        train_num = 60000
        test_num = 10000
        if args.decoder_type == 'bernoulli':
            mnist_train, mnist_test = load_binary_mnist(args.data_dir)
            training_set = torch.from_numpy(mnist_train).float()
            test_set     = torch.from_numpy(mnist_test).float()
        else:
            raise Exception("This implementation only provides Bernoulli decoder for MNIST")

    elif args.dataset == 'ff':
        # Frey-Face dataset (matrix ff of shape (1965, 560))
        dim = 560
        hid_num = 200
        train_num = 1684
        test_num = 281
        ff = scipy.io.loadmat(os.path.join(args.data_dir, 'Frey_Face', 'frey_rawface.mat'))['ff'].T / 256.0
        all_indices = list(range(1965))
        test_idx = random.sample(all_indices, test_num)
        train_idx = list(set(all_indices) - set(test_idx))
        training_set = torch.from_numpy(ff[train_idx, :].astype(np.float32)).float()
        test_set     = torch.from_numpy(ff[test_idx, :].astype(np.float32)).float()
        if args.decoder_type == 'bernoulli':
            raise Exception("Bernoulli decoder not valid for real‐valued Frey-Face")

    elif args.dataset == 'dsprites':
        data_npz = np.load(args.dsprites_path, allow_pickle=True)
        imgs = data_npz['imgs'].astype(np.float32)   # (N, 64,64)
        flat = imgs.reshape(imgs.shape[0], -1)       # (N, 4096)
        dim = flat.shape[1]                          # 4096
        hid_num = 500
        train_num = flat.shape[0]
        test_num = flat.shape[0]
        training_set = torch.from_numpy(flat).float()
        test_set     = torch.from_numpy(flat).float()
        if args.decoder_type != 'bernoulli':
            raise Exception("Only Bernoulli decoder supported on dSprites")
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    print(f"Training set shape: {training_set.shape}")
    print(f"Test set shape:     {test_set.shape}")

    # ─── Prepare DataLoader ─────────────────────────────────────────────────
    train_loader = DataLoader(training_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)

    # ─── Device Selection (CUDA → MPS → CPU) ─────────────────────────────────
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        print(f"Using CUDA device {args.device}: {torch.cuda.get_device_name(args.device)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS backend")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ─── Initialize Model, Optimizer ─────────────────────────────────────────
    net = VAE(args, d=dim, h_num=hid_num)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Ensure save_dir and results/ exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ─── Training + Evaluation Loop ──────────────────────────────────────────
    for epoch in range(args.num_epochs):
        net.set_epoch(epoch)  # for capacity scheduling

        # ─── (1) Training Pass ───────────────────────────────────────────────
        net.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0

        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}", unit="img") as progress_bar:
            for batch in train_loader:
                x = batch.to(device).float()

                optimizer.zero_grad()
                out = net(x)
                if args.decoder_type == 'gaussian':
                    # out = (mu_x, mu_z, logvar_x, logvar_z)
                    loss, recon_loss, kl_loss = net.compute_loss(x, None, out[1], out[3], out[0], out[2])
                else:
                    # out = (x_recon, mu_z, None, logvar_z)
                    loss, recon_loss, kl_loss = net.compute_loss(x, out[0], out[1], out[3])

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_recon += recon_loss.item()
                running_kl += kl_loss.item()

                progress_bar.set_postfix(loss=f"{loss.item():.3f}",
                                         recon=f"{recon_loss.item():.3f}",
                                         kl=f"{kl_loss.item():.3f}")
                progress_bar.update(x.size(0))

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {running_loss / len(train_loader):.3f}, "
            f"Recon: {running_recon / len(train_loader):.3f}, "
            f"KL: {running_kl / len(train_loader):.3f}"
        )

        # ─── (2) Evaluation & Disentanglement Metrics ────────────────────────
        if (epoch + 1) % args.eval_freq == 0:
            net.eval()
            test_elbo = test_function(
                net, test_num,
                dataset=args.dataset,
                out_type=args.decoder_type,
                testset=test_set,
                device=device
            )
            print(f"Epoch {epoch+1}: test ELBO = {test_elbo:.3f}")

            if args.dataset == 'dsprites':
                dsprites_data = DSpritesDataset(args.dsprites_path)
                beta_score = compute_beta_vae_score(dsprites_data, net)
                mig_score = compute_mig(dsprites_data, net)
                print(f"β-VAE score: {beta_score:.3f}, MIG: {mig_score:.3f}")

                # Save latent traversal grid
                traversal_path = os.path.join("results", f"traversals_epoch{epoch+1}.png")
                generate_latent_traversals(net, args.Nz, args.decoder_type,
                                           traversal_path, n_samples=5, size=64)

    # ─── (3) Save Final Model Checkpoint ───────────────────────────────────
    final_filename = (
        f"vae_Nz_{args.Nz}_dataset_{args.dataset}_"
        f"decoder_{args.decoder_type}.pth"
    )
    PATH = os.path.join(args.save_dir, final_filename)
    torch.save(net.state_dict(), PATH)
    print(f"Saved model checkpoint to {PATH}")