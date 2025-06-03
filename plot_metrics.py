import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_log_file(log_path):

    epochs = []
    elbos = []

    recon_epochs = []
    recon_losses = []
    kl_epochs = []
    kl_losses = []

    beta_epochs = []
    beta_scores = []
    mig_epochs = []
    mig_scores = []

    with open(log_path, 'r') as f:
        for line in f:
            # Epoch with  ELBO result
            m = re.search(r'Epoch\s+(\d+): test ELBO =\s*([-+]?[0-9]+\.[0-9]+)', line)
            if m:
                ep = int(m.group(1))
                val = float(m.group(2))
                epochs.append(ep)
                elbos.append(val)
                continue

            # β-VAE score
            m = re.search(r'beta-VAE score:\s*([-+]?[0-9]+\.[0-9]+)', line)
            if m:
                beta_scores.append(float(m.group(1)))
                beta_epochs.append(epochs[-1] if epochs else None)
                continue

            # 3) MIG score
            m = re.search(r'MIG:\s*([-+]?[0-9]+\.[0-9]+)', line)
            if m:
                mig_scores.append(float(m.group(1)))
                mig_epochs.append(epochs[-1] if epochs else None)
                continue

            # Loss: , Recon: , KL:
            m = re.search(
                r'\[Epoch\s+(\d+)\]\s*Loss:\s*([-+]?[0-9]+\.[0-9]+),\s*'
                r'Recon:\s*([-+]?[0-9]+\.[0-9]+),\s*KL:\s*([-+]?[0-9]+\.[0-9]+)',
                line
            )
            if m:
                ep = int(m.group(1))

                recon_epochs.append(ep)
                recon_losses.append(float(m.group(3)))
                kl_epochs.append(ep)
                kl_losses.append(float(m.group(4)))
                continue

    return {
        'epochs':         np.array(epochs),
        'elbos':          np.array(elbos),
        'recon_epochs':   np.array(recon_epochs),
        'recon_losses':   np.array(recon_losses),
        'kl_epochs':      np.array(kl_epochs),
        'kl_losses':      np.array(kl_losses),
        'beta_epochs':    np.array(beta_epochs),
        'beta_scores':    np.array(beta_scores),
        'mig_epochs':     np.array(mig_epochs),
        'mig_scores':     np.array(mig_scores),
    }


def plot_metrics(log_data, output_file):
    with PdfPages(output_file) as pdf:
        # ───  ELBO over epochs ────────────────────────────────────────
        plt.figure(figsize=(8, 5))
        plt.plot(log_data['epochs'], log_data['elbos'], 'b-o', label='Test ELBO')
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')
        plt.title('Test ELBO vs. Epoch')
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # ───  Recon Loss & KL vs. Epoch ───────────────────────────────
        plt.figure(figsize=(8, 5))
        plt.plot(log_data['recon_epochs'], log_data['recon_losses'], 'r-o', label='Reconstruction Loss')
        plt.plot(log_data['kl_epochs'],     log_data['kl_losses'],    'g-o', label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Components')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # ───  Disentanglement Metrics ─────────────────────────────────
        if log_data['beta_scores'].size > 0 and log_data['mig_scores'].size > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(log_data['beta_epochs'], log_data['beta_scores'], 'm-o', label='β-VAE Score')
            plt.plot(log_data['mig_epochs'],  log_data['mig_scores'],  'c-o', label='MIG Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Disentanglement Metrics (β-VAE Score & MIG)')
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot training & disentanglement metrics from a VAE log")
    parser.add_argument('--log', default='results/train.log',
                        help='Path to training log file')
    parser.add_argument('--out', default='training_metrics.pdf',
                        help='Output PDF filename')
    args = parser.parse_args()

    log_data = parse_log_file(args.log)
    plot_metrics(log_data, args.out)
    print(f"Saved plots to {args.out}")