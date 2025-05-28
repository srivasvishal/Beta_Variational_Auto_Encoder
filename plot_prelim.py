import re, argparse, sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# ─── Arg parsing ──────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Plot VAE train/test curves")
p.add_argument('--log', default='results/train.log',
               help='Path to the training log file')
p.add_argument('--out', default='prelim_experiment.pdf',
               help='Output PDF filename')
args = p.parse_args()

train_losses = []
test_elbos   = []

# ─── Read & parse ─────────────────────────────────────────────────────────
with open(args.log, 'r') as f:
    for i, line in enumerate(f, 1):
        line = line.rstrip('\n')
        # train‐loss lines: “[1] loss: 151.583”
        m1 = re.search(r'\[\s*(\d+)\]\s*loss:\s*([0-9]+(?:\.[0-9]+)?)', line)
        if m1:
            train_losses.append(float(m1.group(2)))
        # test‐ELBO lines: “test average ELBO= -137.59”
        m2 = re.search(r'test average ELBO\s*=\s*([-]?[0-9]+(?:\.[0-9]+)?)', line)
        if m2:
            test_elbos.append(float(m2.group(1)))

if not train_losses or not test_elbos:
    print(f"ERROR: parsed zero entries from {args.log}", file=sys.stderr)
    sys.exit(1)

# ─── Plot & save ──────────────────────────────────────────────────────────
with PdfPages(args.out) as pdf:
    # Page 1: Train Loss
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title('Train Loss per Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    pdf.savefig(); plt.close()

    # Page 2: Test ELBO
    plt.figure()
    plt.plot(range(1, len(test_elbos)+1), test_elbos, marker='o')
    plt.title('Test Average ELBO per Epoch')
    plt.xlabel('Epoch'); plt.ylabel('ELBO')
    pdf.savefig(); plt.close()

print(f"Saved {args.out}")