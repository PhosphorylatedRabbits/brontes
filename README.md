# thunder

Thunder simplifies the way you train your pytorch models.
It is based on [pytorch_lightning](https://github.com/williamFalcon/pytorch-lightning)
It comes with an example script in `example/mnist.py` that you can adapt for your needs.

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
base_model = torchvision.models.SqueezeNet(num_classes=10)
```

and use it to wrap your Thunder:

```python
thunder_model = Thunder(
    model=base_model,
    loss=torch.nn.NLLLoss(),
    data_loaders=dataset_loaders,
    optimizers=optimizer
)
```

and finally train the model using `pytorch_lighning`

```python
trainer = pl.Trainer(max_nb_epochs=EPOCHS)
trainer.fit(thunder_model)
```

## Coming soon

Experiment tracking with `mlflow`

## Development setup suggestion

Create a `venv`:

```sh
python -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Install the package in editable mode:

```sh
pip install -e .
```
