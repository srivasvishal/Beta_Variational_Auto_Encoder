# test.py

import numpy as np
import torch
from torch.utils.data import DataLoader

def test_function(model, test_num, dataset, out_type, testset, device):
    """
    Compute average ELBO on the test set.
    Supports Bernoulli or Gaussian decoders.
    """
    model.eval()

    # Wrap the test set (NumPy array or Tensor) in a DataLoader
    if isinstance(testset, np.ndarray):
        data = torch.from_numpy(testset).float()
    else:
        data = testset
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=0)

    total_elbo = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)

            if out_type == 'gaussian':
                # model(x) returns: (mu_x, mu_z, log_sigma_x, log_sigma_z)
                mu_x, mu_z, log_sigma_x, log_sigma_z = model(x)
                # Gaussian log-likelihood term
                recon = -0.5 * torch.sum(
                    (x - mu_x)**2 / torch.exp(log_sigma_x)
                    + log_sigma_x
                    + np.log(2 * np.pi),
                    dim=1
                )
            else:
                # Bernoulli decoder path
                x_recon, mu_z, _, log_sigma_z = model(x)
                recon = torch.sum(
                    x * torch.log(x_recon + 1e-8)
                    + (1 - x) * torch.log(1 - x_recon + 1e-8),
                    dim=1
                )

            # KL divergence term (same for both decoders)
            kl = -0.5 * torch.sum(
                1 + 2 * log_sigma_z
                - mu_z.pow(2)
                - (2 * log_sigma_z).exp(),
                dim=1
            )

            elbo = recon - kl
            total_elbo += elbo.sum().item()

    # Return the average ELBO per data point
    return total_elbo / float(test_num)