import torch
import torch.nn as nn
import numpy as np

class VAE(nn.Module):
    def __init__(self, args, d, h_num, scaled=True):
        super().__init__()
        self.dim = d
        self.Nz = args.Nz
        self.hid_num = h_num
        self.output_type = args.decoder_type
        self.scaled_mean = scaled
        self.beta = args.beta
        self.capacity = args.capacity
        self.current_epoch = 0

        # Encoder
        self.fc1 = nn.Linear(d, h_num)
        self.fc2_mu = nn.Linear(h_num, self.Nz)
        self.fc2_sigma = nn.Linear(h_num, self.Nz)

        # Decoder
        self.fc3 = nn.Linear(self.Nz, h_num)
        if self.output_type == 'gaussian':
            self.fc4_mu = nn.Linear(h_num, d)
            self.fc4_sigma = nn.Linear(h_num, d)
        else:
            self.fc4 = nn.Linear(h_num, d)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def encode(self, x):
        x = x.view(-1, self.dim)
        x = torch.tanh(self.fc1(x))
        mu_z = self.fc2_mu(x)
        log_sigma_z = self.fc2_sigma(x)
        return mu_z, log_sigma_z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = torch.tanh(self.fc3(z))
        if self.output_type == 'gaussian':
            mu = torch.sigmoid(self.fc4_mu(z)) if self.scaled_mean else self.fc4_mu(z)
            log_sigma = self.fc4_sigma(z)
            return mu, log_sigma
        else:
            return torch.sigmoid(self.fc4(z)), None

    def forward(self, x):
        mu_z, log_sigma_z = self.encode(x)
        z = self.reparameterize(mu_z, log_sigma_z)
        if self.output_type == 'gaussian':
            mu_x, log_sigma_x = self.decode(z)
            return mu_x, mu_z, log_sigma_x, log_sigma_z
        else:
            x_recon, _ = self.decode(z)
            return x_recon, mu_z, None, log_sigma_z

    def compute_loss(self, x, recon_x, mu_z, log_sigma_z, mu_x=None, log_sigma_x=None):
        if self.output_type == 'gaussian':
            recon_loss = 0.5 * torch.sum(
                (x - mu_x)**2 / torch.exp(log_sigma_x)
                + log_sigma_x + np.log(2*np.pi),
                dim=1
            )
        else:
            recon_loss = torch.sum(
                x * torch.log(recon_x + 1e-8)
                + (1 - x) * torch.log(1 - recon_x + 1e-8),
                dim=1
            )

        kl_div = -0.5 * torch.sum(
            1 + 2*log_sigma_z - mu_z.pow(2) - (2*log_sigma_z).exp(),
            dim=1
        )

        if self.capacity is not None:
            C = min(self.current_epoch / float(self.capacity), 1.0) * self.beta
            kl_div = torch.abs(kl_div - C)
        else:
            kl_div = self.beta * kl_div

        loss = -torch.mean(recon_loss - kl_div)
        return loss, torch.mean(recon_loss), torch.mean(kl_div)