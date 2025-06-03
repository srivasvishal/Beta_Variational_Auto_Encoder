# Auto-Encoding_Variational_Bayes

A Pytorch Implementation of the paper *Auto-Encoding Variational Bayes* by Diederik P. Kingma and Max Welling.
https://arxiv.org/abs/1312.6114

## Usage

```
usage: main.py [-h] [--dataset {mnist,ff}] [--data_dir DATA_DIR]
               [--epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
               [--device DEVICE] [--save_dir SAVE_DIR]
               [--decoder_type {bernoulli,gaussian}] [--Nz NZ]

Demo for Training VAE

<<<<<<< HEAD
##  MNIST Dataset ---  Input

mkdir -p results

python main.py --dataset mnist --data_dir data --epochs 30 --batch_size 128 --beta 4.0 --capacity 50 --decoder_type bernoulli --Nz 20 --save_dir saved_models --device 0 2>&1 | tee results/train_mnist_capacity.log


## MNIST dataset --report generation

python plot_metrics.py --log results/train_mnist_capacity.log --out results/metrics_mnist_capacity.pdf


##  Dsprites Dataset  --- Input

  python main.py --dataset dsprites --dsprites_path dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --data_dir data --epochs 50 --batch_size 128 --beta 4.0 --capacity 50 --eval_freq 10 --decoder_type bernoulli --Nz 20 --save_dir saved_models --device 0 2>&1 | tee results/train_dsprites_capacity.log

## Dsprites dataset --report generation
python plot_metrics.py --log results/train_dsprites_capacity.log --out results/metrics_dsprites_capacity.pdf

=======


## Datasets 

Binarized MNIST and Frey Face

The binarized MNIST dataset can be downloaded from 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat

The Frey Face dataset can be downloaded from
https://cs.nyu.edu/~roweis/data.html
