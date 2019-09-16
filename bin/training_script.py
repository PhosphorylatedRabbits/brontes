""" Training Script with Thunder Tools """
# imports
import logging
import sys

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
from thunder.core import Thunder

# logging setup
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('FasionMNIST training')

# Configuration parameters
DATA_PATH = 'data/'
BATCH_SIZE = 25
EPOCHS = 10
NUMBER_OF_WORKERS = 1
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

# Thunder Model is initialized with base_model, optimizer, loss, data_loaders
# Optionally a dict of metrics functions and a batch_fn applied to every batch
thunder_model = Thunder(
    model=base_model,
    loss=torch.nn.NLLLoss(),
    data_loaders=dataset_loaders,
    optimizers=optimizer
)

# Finally train the model
trainer = pl.Trainer(max_nb_epochs=EPOCHS)
trainer.fit(thunder_model)
