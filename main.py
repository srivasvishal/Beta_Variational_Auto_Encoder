# This the Python script to train the VAE model.

import torch
import numpy as np
from vae_net import VAE
import argparse
from tqdm import tqdm
import time
import scipy.io
import random
import math
from torch.utils.data import Dataset, DataLoader
from utils.test import test_function

class DSpritesDataset(Dataset):
    """Loader for the dSprites npz file (key 'imgs', shape N×64×64)."""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.imgs = data['imgs'].astype(np.float32)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx: int):
        # return a (1×64×64) tensor for each sprite
        return torch.from_numpy(self.imgs[idx]).unsqueeze(0)
    
    
    
def load_binary_mnist(d_dir):
    train_file = d_dir + '/BinaryMNIST/binarized_mnist_train.amat'
    valid_file = d_dir + '/BinaryMNIST/binarized_mnist_valid.amat'
    test_file = d_dir + '/BinaryMNIST/binarized_mnist_test.amat'
    mnist_train = np.concatenate([np.loadtxt(train_file), np.loadtxt(valid_file)])
    mnist_test = np.loadtxt(test_file)
    return mnist_train, mnist_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Training VAE")
    parser.add_argument("--dataset", default='mnist', dest='dataset',
                        choices=('mnist', 'ff', 'dsprites'),
                        help="Dataset to train the VAE")
    parser.add_argument("--dsprites_path", dest='dsprites_path',
                        default="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
                        help="Path to the dSprites .npz file")
    parser.add_argument("--data_dir", dest='data_dir', default="../../dataset",
                        help="The directory of your dataset")
    parser.add_argument("--epochs", dest='num_epochs', default=5, type=int,
                        help="Total number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=100, type=int,
                        help="The batch size")
    parser.add_argument('--device', dest='device', default=0, type=int,
                        help='Index of device')
    parser.add_argument("--save_dir", dest='save_dir', default="./saved_models",
                        help="The directory to save your trained model")
    parser.add_argument('--decoder_type', dest='decoder_type', default='bernoulli', type=str,
                        help='Type of your decoder', choices=('bernoulli', 'gaussian'))
    parser.add_argument("--Nz", default=20, type=int,
                        help="Nz (dimension of the latent code)")
    args = parser.parse_args()

    # load data
    if args.dataset == 'mnist':
        dim = 28 * 28
        hid_num = 500
        train_num = 60000
        test_num = 10000
        if args.decoder_type == 'bernoulli':
            training_set, test_set = load_binary_mnist(args.data_dir)
        else:
            raise Exception("This implementation only provides Bernoulli decoder for MNIST")
    elif args.dataset == 'ff':
        dim = 560
        hid_num = 200
        train_num = 1684
        test_num = 281
        ff = scipy.io.loadmat(args.data_dir + '/Frey_Face/frey_rawface.mat')['ff'].transpose() / 256
        test_index = random.sample(list(range(1965)), 281)
        train_index = list(set(range(1965)) - set(test_index))
        training_set = ff[train_index, :]
        test_set = ff[test_index, :]
        if args.decoder_type == 'bernoulli':
            raise Exception("Can't use Bernoulli decoder on Frey Face")

    elif args.dataset == 'dsprites':
        data = np.load(args.dsprites_path)
        imgs = data['imgs'].astype(np.float32)             # (N, 64, 64)
        flat = imgs.reshape(imgs.shape[0], -1)            # (N, 4096)

        # Mirror the MNIST branch: two arrays for train/test
        training_set = flat                               # shape (N, 4096)
        test_set     = flat                               # you can split if you want

        # Set the hyperparameters the same way as MNIST/Frey
        dim       = flat.shape[1]                         # 4096
        hid_num   = 500                                   
        train_num = training_set.shape[0]                 # N
        test_num  = test_set.shape[0]                     # N

        # only Bernoulli decoder makes sense for binary sprites
        if args.decoder_type != 'bernoulli':
            raise Exception("Only Bernoulli decoder supported on dSprites")

    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    print(training_set.shape)
    print(test_set.shape)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    # define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = VAE(args, d=dim, h_num=hid_num)
    net.to(device)
    optimizer = torch.optim.Adagrad(net.parameters())

    # train the model
    start = time.time()
    for epoch in range(args.num_epochs):
        # test
        if epoch % 10 == 1:
            test_elbo = test_function(net, test_num, dataset=args.dataset, out_type=args.decoder_type,
                                      testset=test_set, device=device)
            print('test average ELBO=', test_elbo)

        # iterations
        running_loss = 0.0
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for i, data in enumerate(train_loader, 0):
                train = data.to(device)
                optimizer.zero_grad()

                output = net(train.float())

                # the negative KL term
                negative_KL = (torch.ones_like(output[1]) + 2 * output[3] - output[1] * output[1] - torch.exp(
                    2 * output[3])).sum(1) / 2

                # the log conditional prob term
                if args.decoder_type == 'gaussian':
                    train_minus_mu = train - output[0]
                    log_p_x_given_z = -torch.ones_like(train).sum(1) * np.log(2 * math.pi) / 2 \
                                      - output[2].sum(1) / 2 - (
                                              train_minus_mu * train_minus_mu / (2 * torch.exp(output[2]))).sum(1)
                else:
                    log_p_x_given_z = torch.sum(output[0] * train - torch.log(1 + torch.exp(output[0])), 1)

                # update parameters
                loss = -negative_KL.mean() - log_p_x_given_z.mean()
                loss.backward()
                optimizer.step()
                running_loss -= negative_KL.sum().item()
                running_loss -= log_p_x_given_z.sum().item()

                # progress bar
                progress_bar.set_postfix(loss=loss.mean().item())
                progress_bar.update(data.size(0))

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / train_num))

    print('Finished Training, time cost', time.time() - start)

    PATH = args.save_dir + '/vae_Nz_' + str(
        args.Nz) + '_dataset_' + args.dataset + '_decoder_' + args.decoder_type + '.pth'
    torch.save(net.state_dict(), PATH)
