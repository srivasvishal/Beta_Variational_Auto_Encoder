import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import torch


def compute_beta_vae_score(dataset, model, num_samples=10000, random_state=42):
    factors = dataset.sample_factors(num_samples, random_state)
    observations = dataset.sample_observations_from_factors(factors, random_state)  # Tensor

    device = next(model.parameters()).device
    observations = observations.to(device)

    with torch.no_grad():
        mus, _ = model.encode(observations)
    mus_np = mus.cpu().numpy()

    mus_np = (mus_np - mus_np.mean(axis=0)) / (mus_np.std(axis=0) + 1e-8)

    rng = np.random.RandomState(random_state)
    factor_id = rng.randint(factors.shape[1], size=num_samples)
    z_diff = np.zeros((num_samples, mus_np.shape[1]), dtype=np.float32)

    for i in range(num_samples):
        k = factor_id[i]
        same_factor = np.where(factors[:, k] == factors[i, k])[0]
        j = rng.choice(same_factor)
        z_diff[i] = np.abs(mus_np[i] - mus_np[j])

    X_train, X_test, y_train, y_test = train_test_split(
        z_diff, factor_id, test_size=0.2, random_state=random_state
    )

    clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc


def compute_mig(dataset, model, num_samples=10000, random_state=42):

    factors = dataset.sample_factors(num_samples, random_state)
    observations = dataset.sample_observations_from_factors(factors, random_state)
    device = next(model.parameters()).device
    observations = observations.to(device)


    with torch.no_grad():
        mus, _ = model.encode(observations)
    mus_np = mus.cpu().numpy()
    B = 20
    num_latents = mus_np.shape[1]  # Nz Value
    z_disc = np.zeros_like(mus_np, dtype=int)

    for j in range(num_latents):
        # Compute bin edges for the j-th latent dimension
        edges = np.histogram_bin_edges(mus_np[:, j], bins=B)
        # Digitize: values in [0..B-1]
        z_disc[:, j] = np.digitize(mus_np[:, j], edges[:-1])

    num_factors = factors.shape[1]  # for dSprites, this is 5
    mi_matrix = np.zeros((num_factors, num_latents), dtype=np.float32)

    for k in range(num_factors):
        for j in range(num_latents):
            mi_matrix[k, j] = mutual_info_score(factors[:, k], z_disc[:, j])

    factor_entropies = np.zeros(num_factors, dtype=np.float32)
    for k in range(num_factors):
        counts = np.bincount(factors[:, k])
        factor_entropies[k] = entropy(counts + 1e-8)

    normalized_mi = mi_matrix / np.expand_dims(factor_entropies, axis=1)

    sorted_mi = np.sort(normalized_mi, axis=1)[:, ::-1]
    top1 = sorted_mi[:, 0]
    top2 = sorted_mi[:, 1]

    mig_score = np.mean(top1 - top2)
    return mig_score