[![Build Status](https://travis-ci.org/PhosphorylatedRabbits/brontes.svg?branch=master)](https://travis-ci.org/PhosphorylatedRabbits/brontes)
[![codecov](https://codecov.io/gh/PhosphorylatedRabbits/brontes/branch/master/graph/badge.svg)](https://codecov.io/gh/PhosphorylatedRabbits/brontes)
[![Updates](https://pyup.io/repos/github/PhosphorylatedRabbits/brontes/shield.svg)](https://pyup.io/repos/github/PhosphorylatedRabbits/brontes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/brontes.svg)](https://badge.fury.io/py/brontes)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6dc4922e69954e96820dae93ed704736)](https://www.codacy.com/manual/drugilsberg/brontes?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=PhosphorylatedRabbits/brontes&amp;utm_campaign=Badge_Grade)

# Brontes

Brontes is your helping cyclops for pytorch models training.
It is based on [pytorch_lightning](https://github.com/williamFalcon/pytorch-lightning)
and comes with an example script in [examples/mnist/run.py](examples/mnist/run.py) that you can adapt for your needs.

Additionally, there is an example in [examples/mlflow](examples/mlflow) which describes how to use mlflow with Brontes.

## Just do this

Define your dataset_loaders as a dictionary: use `train`, `val` and optionally a `test` split:

```python
dataset_loaders = {
    'train':
        torch.utils.data.DataLoader(
            datasets.MNIST(
                root=DATA_PATH,
                train=True,
                download=True
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
                download=True
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUMBER_OF_WORKERS
        )
}
```

define your acrhitecture as a `torch.nn.Module` (or pick an existing architecture):

```python
base_model = brontes.examples.Net()
```

and wrap it with Brontes:

```python
brontes_model = Brontes(
    model=base_model,
    loss=torch.nn.NLLLoss(),
    data_loaders=dataset_loaders,
    optimizers=optimizer
)
```

finally train the model using `pytorch_lighning`

```python
trainer = pl.Trainer(max_nb_epochs=EPOCHS)
trainer.fit(brontes_model)
```

## Development setup

Setup the conda environment

```sh
conda env create -f conda.yml
```

Activate it:

```sh
conda activate brontes
```
