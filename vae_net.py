

import torch
import torch.nn as nn
import numpy as np


class VAE(nn.Module):
    def __init__(self, args, d, h_num, scaled=True):
        super(VAE, self).__init__()
        self.dim = d
        self.Nz = args.Nz
        self.hid_num = h_num
        self.output_type = args.decoder_type
        self.scaled_mean = scaled

        # β parameter is initialise here : which is baseline paper's implementation
        self.beta = args.beta if hasattr(args, 'beta') else 1.0
        # capacity scheduling: number of epochs to reach full KL capacity  :  3rd reference Paper
        self.capacity = args.capacity if hasattr(args, 'capacity') else None
        self.current_epoch = 0

        # ─── Encoder ──────────────────────────────────────────────────────────
        # Single hidden layer
        self.fc1 = nn.Linear(d, h_num)
        self.fc2_mu = nn.Linear(h_num, args.Nz)
        self.fc2_logvar = nn.Linear(h_num, args.Nz)

        # ─── Decoder ──────────────────────────────────────────────────────────
        # Single hidden layer
        self.fc3 = nn.Linear(args.Nz, h_num)
        if args.decoder_type == 'gaussian':
            self.fc4_mu = nn.Linear(h_num, d)
            self.fc4_logvar = nn.Linear(h_num, d)
        else:
            self.fc4 = nn.Linear(h_num, d)

    def set_epoch(self, epoch):

        self.current_epoch = epoch

    def encode(self, x):

        ###  Important Part : Need to review again #______________________________________________________

        """
        Encode input x into latent parameters (μ_z, logσ²_z).
        x: tensor of shape (batch_size, dim)
        Returns: (mu_z, logvar_z) each of shape (batch_size, Nz)
        """
        x = x.view(-1, self.dim)
        h = torch.tanh(self.fc1(x))
        mu_z = self.fc2_mu(h)
        logvar_z = self.fc2_logvar(h)
        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ∼ N(mu, exp(logvar)).
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(½·logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):

        ###  Important Part : Need to review again #______________________________________________________

        """
        Decode latent z into reconstruction.
        If Gaussian decoder: returns (mu_x, logvar_x).
        If Bernoulli decoder: returns (p_x, None).
        """
        h = torch.tanh(self.fc3(z))
        if self.output_type == 'gaussian':
            if self.scaled_mean:
                mu_x = torch.sigmoid(self.fc4_mu(h))
            else:
                mu_x = self.fc4_mu(h)
            logvar_x = self.fc4_logvar(h)
            return mu_x, logvar_x
        else:
            px = torch.sigmoid(self.fc4(h))
            return px, None

    def forward(self, x):


        ###  Important Part : Need to review again #______________________________________________________


        """
        # Full forward pass: x → encode → reparameterize → decode.
        Returns:
          - If Gaussian: (mu_x, mu_z, logvar_x, logvar_z)
          - If Bernoulli: (x_recon, mu_z, None, logvar_z)
        """
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)

        if self.output_type == 'gaussian':
            mu_x, logvar_x = self.decode(z)
            return mu_x, mu_z, logvar_x, logvar_z
        else:
            x_recon, _ = self.decode(z)
            return x_recon, mu_z, None, logvar_z

    def compute_loss(self, x, recon_x, mu_z, logvar_z, mu_x=None, logvar_x=None):

        batch_size = x.size(0)

        # ─── Reconstruction Loss ─────────────────────────────────────────────
        if self.output_type == 'gaussian':
            # Negative log-likelihood of Gaussian
            recons = 0.5 * torch.sum(
                (x - mu_x) ** 2 / torch.exp(logvar_x)
                + logvar_x
                + np.log(2 * np.pi),
                dim=1
            )
        else:
            # Bernoulli log-likelihood:
            recons = torch.sum(
                x * torch.log(recon_x.clamp(min=1e-8))
                + (1 - x) * torch.log((1 - recon_x).clamp(min=1e-8)),
                dim=1
            )


        # ─── KL Divergence ─────────────────────────────────────────────
        kl = -0.5 * torch.sum(
            1
            + logvar_z
            - mu_z.pow(2)
            - torch.exp(logvar_z),
            dim=1
        )

        # ─── Apply β or Capacity Scheduling ─────────────────────────────────
        if self.capacity is not None:
            # capacity = number of epochs to reach full weight = β
            C_t = min(self.current_epoch / float(self.capacity), 1.0) * self.beta
            kl_term = torch.abs(kl - C_t)
        else:
            # fixed‐β VAE
            kl_term = self.beta * kl

        # ─── Total Loss ────────
        elbo = recons - kl_term
        loss = -torch.mean(elbo) r


        return loss, torch.mean(-recons), torch.mean(kl_term)