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

python main.py \
  --dataset      mnist \
  --data_dir     data/binarized_mnist \
  --epochs       50 \
  --batch_size   64 \
  --decoder_type bernoulli \
  --Nz           10 \
  --save_dir     results \
2>&1 | tee results/train.log


## MNIST dataset --report generation

python plot_prelim.py \
  --log  results/train.log \
  --out  prelim_experiment_mnist.pdf


##  Dsprites Dataset  --- Input

mkdir -p results/dsprites

python main.py \
  --dataset      dsprites \
  --dsprites_path dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
  --epochs       50 \
  --batch_size   64 \
  --decoder_type bernoulli \
  --Nz           10 \
  --device       0 \
  --save_dir     results/dsprites \
2>&1 | tee results/dsprites/train.log

## Dsprites dataset --report generation
python plot_prelim.py \
  --log results/train.log \
  --out  prelim_experiment_dsprites.pdf

=======
optional arguments:
  -h, --help            show this help message and exit
  --dataset {mnist,ff}  Dataset to train the VAE
  --data_dir DATA_DIR   The directory of your dataset
  --epochs NUM_EPOCHS   Total number of epochs
  --batch_size BATCH_SIZE
                        The batch size
  --device DEVICE       Index of device
  --save_dir SAVE_DIR   The directory to save your trained model
  --decoder_type {bernoulli,gaussian}
                        Type of your decoder
  --Nz NZ               Nz (dimension of the latent code)
>>>>>>> 2925d8f (Initial changes for Beta-VAE)
```

It can be seen by running
```
python main.py --help
```

## Datasets 

Binarized MNIST and Frey Face

The binarized MNIST dataset can be downloaded from 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat

The Frey Face dataset can be downloaded from
https://cs.nyu.edu/~roweis/data.html
