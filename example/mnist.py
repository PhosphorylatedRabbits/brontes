""" Training Script with Thunder Tools """
# imports
import logging
import sys

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from brontes import Brontes

import mlflow

# logging setup
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('MNIST training')

# Configuration parameters
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
    '-n', '--model_name', type=int,
    help='model name.', default=None,
    required=False
)

parser.add_argument(
    '-s', '--seed', type=int,
    help='seed for reproducible results.', default=42,
    required=False
)

parser.add_argument(
    '-b', '--batch_size', type=int,
    help='batch size.', default=25,
    required=False
)

parser.add_argument(
    '--epochs', type=int,
    help='epochs.', default=5,
    required=False
)

parser.add_argument(
    '-l', '--learning_rate', type=float,
    help='learning rate.', default=1e-5,
    required=False
)

arguments = parser.parse_args()
DATA_PATH = arguments.data_path
NUMBER_OF_WORKERS = arguments.number_of_workers
MODEL_NAME = arguments.model_name
SEED = arguments.seed
BATCH_SIZE = arguments.batch_size
EPOCHS = arguments.epochs
LEARNING_RATE = arguments.learning_rate


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # data loaders for the MNIST dataset
    dataset_loaders = {
        'train':
            torch.utils.data.DataLoader(
                datasets.MNIST(
                    root=DATA_PATH,
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                        ]
                    )
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUMBER_OF_WORKERS
            ),
        'val':
            torch.utils.data.DataLoader(
                datasets.MNIST(
                    root=DATA_PATH,
                    train=False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                        ]
                    )
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUMBER_OF_WORKERS
            )
    }

    # Definition of Base Model
    base_model = torch.nn.Sequential(
        models.SqueezeNet(num_classes=10)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5  # standard value
    )

    # Brontes Model is initialized with base_model, optimizer, loss, data_loaders
    # Optionally a dict of metrics functions and a batch_fn applied to every batch
    brontes_model = Brontes(
        model=base_model,
        loss=torch.nn.NLLLoss(),
        data_loaders=dataset_loaders,
        optimizers=optimizer,
        training_log_interval=10
    )
    

    # Finally train the model
    trainer = pl.Trainer(max_nb_epochs=EPOCHS)
    trainer.fit(brontes_model)


if __name__ == "__main__":
    main()
