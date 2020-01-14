"""Training script with brontes."""
# imports
import logging
import sys
import argparse
import tempfile
import os
import numpy as np

import torch
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

from brontes.examples import VAE
from brontes import Brontes

# logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('vae-mnist-training')

# # configure argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--data_path', type=str,
    help='path to the data.', default='data/',
    required=False
)
parser.add_argument(
    '-p', '--number_of_workers', type=int,
    help='number of workers.', default=1,
    required=False
)
parser.add_argument(
    '-n', '--model_name', type=str,
    help='model name.', default='model',
    required=False
)
parser.add_argument(
    '-s', '--seed', type=int,
    help='seed for reproducible results.', default=42,
    required=False
)
parser.add_argument(
    '-b', '--batch_size', type=int,
    help='batch size.', default=32,
    required=False
)
parser.add_argument(
    '--epochs', type=int,
    help='epochs.', default=2,
    required=False
)
parser.add_argument(
    '-l', '--learning_rate', type=float,
    help='learning rate.', default=10e-3,
    required=False
)


def main(args):
    """
    Train VAE on MNIST with brontes.

    Args:
        args (Namespace): parsed args.
    """

    # make sure the data folder exists
    os.makedirs(args.data_path, exist_ok=True)

    # set the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # data loaders for the MNIST dataset
    def collate_fn(batch):
        """
        In a VAE the input and the target are the same.
        """
        x = torch.cat([example[0] for example in batch])
        return x, x

    dataset_loaders = {
        'train':
            torch.utils.data.DataLoader(
                datasets.MNIST(
                    root=args.data_path,
                    train=True,
                    download=True,
                    transform=transforms.ToTensor()
                ),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.number_of_workers,
                collate_fn=collate_fn
            ),
        'val':
            torch.utils.data.DataLoader(
                datasets.MNIST(
                    root=args.data_path,
                    train=False,
                    download=True,
                    transform=transforms.ToTensor()
                ),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.number_of_workers,
                collate_fn=collate_fn
            )
    }

    # definition of base model
    vae_model = VAE()

    optimizer = torch.optim.Adam(
        vae_model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5  # standard value
    )

    def loss_fn(out, x):
        recon_x, mu, logvar = out
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    brontes_model = Brontes(
        model=vae_model,
        loss=loss_fn,
        data_loaders=dataset_loaders,
        optimizers=optimizer,
        training_log_interval=10
    )

    # finally, train the model
    trainer = pl.Trainer(max_nb_epochs=args.epochs)
    trainer.fit(brontes_model)

    # save the model
    saved_model = f'{tempfile.mkdtemp()}/{args.model_name}.pt'
    torch.save(brontes_model.model, saved_model)


if __name__ == "__main__":
    main(args=parser.parse_args())
