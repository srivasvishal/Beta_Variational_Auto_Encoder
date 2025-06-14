import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def parse_log_file(log_path):
    epochs, elbos, recon_losses, kl_losses = [], [], [], []
    beta_scores, mig_scores = [], []

    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'Epoch (\d+): test ELBO = ([-+]?\d*\.\d+)', line)
            if m:
                epochs.append(int(m.group(1)))
                elbos.append(float(m.group(2)))
                continue
            m = re.search(r'β-VAE score: ([-+]?\d*\.\d+)', line)
            if m:
                beta_scores.append(float(m.group(1)))
                continue
            m = re.search(r'MIG: ([-+]?\d*\.\d+)', line)
            if m:
                mig_scores.append(float(m.group(1)))
                continue
            m = re.search(r'Loss: ([-+]?\d*\.\d+), Recon: ([-+]?\d*\.\d+), KL: ([-+]?\d*\.\d+)', line)
            if m:
                recon_losses.append(float(m.group(2)))
                kl_losses.append(float(m.group(3)))

    return {
        'epochs': np.array(epochs),
        'elbos': np.array(elbos),
        'recon_losses': np.array(recon_losses),
        'kl_losses': np.array(kl_losses),
        'beta_scores': np.array(beta_scores),
        'mig_scores': np.array(mig_scores)
    }

def plot_metrics(log_data, output_file):
    with PdfPages(output_file) as pdf:
        # ELBO
        plt.figure(figsize=(8, 5))
        plt.plot(log_data['epochs'], log_data['elbos'], label='ELBO')
        plt.xlabel('Epoch'); plt.ylabel('ELBO')
        plt.title('Test ELBO over Training'); plt.grid(True)
        pdf.savefig(); plt.close()

        # Loss components
        plt.figure(figsize=(8, 5))
        plt.plot(log_data['recon_losses'], label='Recon Loss')
        plt.plot(log_data['kl_losses'],  label='KL Divergence')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('Training Loss Components'); plt.legend(); plt.grid(True)
        pdf.savefig(); plt.close()

        # Disentanglement metrics (if any)
        if len(log_data['beta_scores']) or len(log_data['mig_scores']):
            plt.figure(figsize=(8, 5))
            if len(log_data['beta_scores']):
                plt.plot(log_data['epochs'][:len(log_data['beta_scores'])],
                         log_data['beta_scores'], label='β-VAE Score')
            if len(log_data['mig_scores']):
                plt.plot(log_data['epochs'][:len(log_data['mig_scores'])],
                         log_data['mig_scores'], label='MIG Score')
            plt.xlabel('Epoch'); plt.ylabel('Score')
            plt.title('Disentanglement Metrics'); plt.legend(); plt.grid(True)
            pdf.savefig(); plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='results/train.log')
    parser.add_argument('--out', default='training_metrics.pdf')
    args = parser.parse_args()

    data = parse_log_file(args.log)
    plot_metrics(data, args.out)