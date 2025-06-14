#!/usr/bin/env python3
"""
Standalone script to generate latent traversals from a saved VAE model.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from vae_net import VAE


def generate_latent_traversals(model, n_z, out_type, filename, n_samples=10, size=28, dataset='mnist'):
    """Generate latent traversal visualizations."""
    model.eval()
    device = next(model.parameters()).device
    
    # Use multiple starting points for more diverse traversals
    num_starting_points = 3 if dataset == 'mnist' else 1
    
    all_grid_imgs = []
    
    for start_idx in range(num_starting_points):
        torch.manual_seed(start_idx * 42)  # For reproducible results
        z0 = torch.randn(1, n_z, device=device)
        
        grid_imgs = []
        with torch.no_grad():
            for dim in range(n_z):
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
        
        all_grid_imgs.extend(grid_imgs)

    # Create the final grid
    grid = torch.cat(all_grid_imgs, dim=0)
    
    if dataset == 'mnist' and num_starting_points > 1:
        grid_img = make_grid(grid, nrow=n_samples, normalize=True, pad_value=1.0)
        plt.figure(figsize=(n_samples * 1.5, n_z * num_starting_points * 0.8))
    else:
        grid_img = make_grid(grid, nrow=n_samples, normalize=True, pad_value=1.0)
        plt.figure(figsize=(n_samples * 1.2, n_z * 0.8))

    grid_np = grid_img.permute(1, 2, 0).cpu().numpy()
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
    
    plt.imshow(grid_np, cmap='gray')
    plt.axis('off')
    plt.title(f'Latent Traversals - {dataset.upper()}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    model.train()
    print(f"Saved latent traversals to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate latent traversals from saved VAE model")
    parser.add_argument("--model_path", required=True, help="Path to saved model (.pth file)")
    parser.add_argument("--dataset", default='mnist', choices=('mnist', 'ff', 'dsprites'),
                        help="Dataset the model was trained on")
    parser.add_argument("--decoder_type", default='bernoulli', choices=('bernoulli', 'gaussian'),
                        help='Decoder type')
    parser.add_argument("--Nz", default=20, type=int, help="Latent dimension")
    parser.add_argument("--output", default="latent_traversals.png", help="Output filename")
    parser.add_argument("--n_samples", default=10, type=int, help="Samples per dimension")
    parser.add_argument('--device', default=0, type=int, help='CUDA device (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Device setup
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    # Dataset-specific parameters
    if args.dataset == 'mnist':
        dim = 28 * 28
        hid_num = 500
        size = 28
    elif args.dataset == 'ff':
        dim = 560
        hid_num = 200
        size = int(np.sqrt(560))  # approximately 24
    elif args.dataset == 'dsprites':
        dim = 64 * 64
        hid_num = 500
        size = 64
    
    # Create a temporary args object for VAE initialization
    class TempArgs:
        def __init__(self):
            self.Nz = args.Nz
            self.decoder_type = args.decoder_type
            self.beta = 1.0
            self.capacity = None
    
    temp_args = TempArgs()
    
    # Load model
    model = VAE(temp_args, d=dim, h_num=hid_num)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generate traversals
    generate_latent_traversals(
        model, args.Nz, args.decoder_type, args.output,
        n_samples=args.n_samples, size=size, dataset=args.dataset
    )
    
    print(f"Latent traversals saved to {args.output}")