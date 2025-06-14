import numpy as np
import torch
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

def compute_beta_vae_score(dataset, model, num_samples=10000, random_state=42):
    """
    β-VAE disentanglement score (dSprites only).
    Sample full 6‐factor tuples, then drop the constant color factor (size=1).
    """
    # 1) sample full tuples
    full_factors = dataset.sample_factors(num_samples, random_state)  # (N,6)
    obs = dataset.sample_observations_from_factors(full_factors, random_state)

    # 2) encode & normalize
    device = next(model.parameters()).device
    obs = obs.to(device)
    with torch.no_grad():
        mu_z, _ = model.encode(obs)
    mus = mu_z.cpu().numpy()
    mus = (mus - mus.mean(axis=0)) / (mus.std(axis=0) + 1e-8)

    # 3) drop the color factor (first element where latents_sizes==1)
    keep = [size > 1 for size in dataset.latents_sizes]
    factors = full_factors[:, keep]  # now (N,5)

    # 4) build z_diff & labels
    factor_id = np.random.randint(factors.shape[1], size=num_samples)
    z_diff = np.zeros((num_samples, mus.shape[1]), dtype=np.float32)
    for i in range(num_samples):
        k = factor_id[i]
        same = np.where(factors[:, k] == factors[i, k])[0]
        j = np.random.choice(same)
        z_diff[i] = np.abs(mus[i] - mus[j])

    # 5) train & evaluate classifier
    Xtr, Xte, ytr, yte = train_test_split(
        z_diff, factor_id, test_size=0.2, random_state=random_state
    )
    clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte)


def compute_mig(dataset, model, num_samples=10000, random_state=42):
    """
    Mutual Information Gap (dSprites only).
    Sample full 6‐factor tuples, drop the color factor, then compute MIG.
    """
    full_factors = dataset.sample_factors(num_samples, random_state)  # (N,6)
    obs = dataset.sample_observations_from_factors(full_factors, random_state)

    device = next(model.parameters()).device
    obs = obs.to(device)
    with torch.no_grad():
        mu_z, _ = model.encode(obs)
    mus = mu_z.cpu().numpy()

    # drop color factor
    keep = [size > 1 for size in dataset.latents_sizes]
    factors = full_factors[:, keep]  # (N,5)

    num_factors, Nz = factors.shape[1], mus.shape[1]

    # discretize latents into B bins
    B = 20
    z_disc = np.zeros_like(mus, dtype=int)
    for j in range(Nz):
        _, edges = np.histogram(mus[:, j], bins=B)
        z_disc[:, j] = np.digitize(mus[:, j], edges[:-1], right=False)

    # mutual information matrix
    mi = np.zeros((num_factors, Nz), dtype=np.float64)
    for k in range(num_factors):
        for j in range(Nz):
            mi[k, j] = mutual_info_score(factors[:, k], z_disc[:, j])

    # entropies of each factor
    ent = np.zeros(num_factors, dtype=np.float64)
    for k in range(num_factors):
        counts = np.bincount(factors[:, k])
        ent[k] = entropy(counts + 1e-8)

    # normalized MI and compute gap
    norm_mi = mi / ent.reshape(num_factors, 1)
    sorted_mi = -np.sort(-norm_mi, axis=1)
    mig_score = float(np.mean(sorted_mi[:, 0] - sorted_mi[:, 1]))
    return mig_score