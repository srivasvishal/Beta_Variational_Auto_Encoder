By diving into Variational Autoencoders (VAE) and β-VAEs, we explore the
effects β has on disentanglement of latent expressions. Using MNIST and dSprites
datasets, we evaluate the models impacts on reconstruction quality, and latent space
structure. Our results demonstrate that moderate β values promote meaningful
disentanglement, balancing the trade-off between fidelity and factorization.

```


Demo for Training VAE


##  MNIST Dataset ---  Input

mkdir -p results

python main.py --dataset mnist --data_dir data --epochs 30 --batch_size 128 --beta 4.0 --capacity 50 --decoder_type bernoulli --Nz 20 --save_dir saved_models --device -1 2>&1 | tee results/train_mnist_capacity.log


## MNIST dataset --  report generation

{ MNIST has no ground‐truth “latents” (no shape/scale/rotation/position factors), so we don’t attempt to compute β‐VAE score or MIG .}

python plot_metrics.py --log results/train_mnist_capacity.log --out results/metrics_mnist_capacity.pdf


##  Dsprites Dataset  --- Input

  python main.py --dataset dsprites --dsprites_path dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --data_dir data --epochs 50 --batch_size 128 --beta 4.0 --capacity 50 --eval_freq 10 --decoder_type bernoulli --Nz 20 --save_dir saved_models --device -1 2>&1 | tee results/train_dsprites_capacity.log

## Dsprites dataset --  report generation

python plot_metrics.py --log results/train_dsprites_capacity.log --out results/metrics_dsprites_capacity.pdf


## Frey face  Dataset  --- Input

{ Frey‐Face has no ground‐truth “latents” (no shape/scale/rotation/position factors), so we don’t attempt to compute β‐VAE score or MIG .}

python main.py --dataset ff --data_dir data --epochs 30 --batch_size 64 --beta 4.0 --capacity None --decoder_type gaussian --Nz 20 --save_dir saved_models --device 0 2>&1 | tee results/train_freyface_beta4.log

## Frey face  Dataset ---  report generation

python plot_metrics.py --log results/train_freyface_beta4.log --out results/metrics_freyface_beta4.pdf


=======



