"""Brontes training utilities."""
import torch
import pytorch_lightning as pl
from .tracker import Tracker


class Brontes(pl.LightningModule):
    """
    Brontes is a class to facilitate the write-up and reduce
    boilerplate code in training scripts.
    """

    def __init__(
        self, model, loss,
        data_loaders, optimizers,
        metrics={}, batch_fn=None
    ):
        """
        Initialize a thuhnder instance.

        Args:
            model (torch.nn.Module): a model object.
            loss (torch.nn.Module): a loss object.
            data_loaders (dict): a dict of torch.utils.data.DataLoader objects.
                It has to contain 'train', 'val' and optionally a 'test'.
            optimizers (list of/or torch.optim.Optimizer): optimizer/s adopted.
            metrics (dict): additional metrics to compute apart from the loss.
                Defaults to {}.
            batch_fn (function): a function to preprocess the batch.
                Defaults to None, the identity.
        """
        super(Brontes, self).__init__()
        self.model = model
        self.loss = loss
        self.data_loaders = data_loaders
        self.tracker = Tracker()
        # make sure we have the needed data loaders.
        if not (
            'train' in self.data_loaders and
            'val' in self.data_loaders
        ):
            raise RuntimeError(
                'Argument "data_loaders" must contain '
                'both "train" and "val" keys.'
            )
        self.optimizers = optimizers
        self.metrics = metrics
        if batch_fn is not None:
            self.batch_fn = batch_fn
        else:
            self.batch_fn = lambda x: x

    def forward(self, x):
        """
        Forward pass in the model.

        Args:
            x (torch.tensor): a tensor.

        Returns:
            a torch.tensor.
        """
        return self.model.forward(x)

    def training_step(self, batch, batch_nb):
        """
        Training step.

        Args:
            batch (tuple): a tuple containing x and y.
            batch_nb (int): integer representing the batch number.

        Returns:
            a dict containing the loss and, optionally, additional metrics.
        """
        x, y = self.batch_fn(batch)
        y_hat = self.forward(x)
        training_dict = {}
        training_dict['loss'] = self.loss(y_hat, y)
        for name, metric in self.metrics.items():
            training_dict[name] = metric(y_hat, y)     
        self.tracker.log_tensor_dict(training_dict)
        return training_dict

    def validation_step(self, batch, batch_nb):
        """
        Validation step.

        Args:
            batch (tuple): a tuple containing x and y.
            batch_nb (int): integer representing the batch number.

        Returns:
            a dict containing the loss and, optionally, additional metrics.
        """
        validation_dict = {
            f'val_{name}': value
            for name, value in self.training_step(batch, batch_nb).items()
        }
        self.tracker.log_tensor_dict(validation_dict)
        return validation_dict

    def validation_end(self, outputs):
        """
        Validation end.

        Args:
            outputs (iterable): an iterable containing results.

        Returns:
            a dict containing the loss and, optionally, additional metrics
            averaged over the results.
        """
        names = ['val_loss'] + [f'val_{name}' for name in self.metrics.keys()]
        validation_end_dict = {
            f'avg_{name}': torch.stack([x[name] for x in outputs]).mean()
            for name in names
        }
        self.tracker.log_tensor_dict(validation_end_dict)
        return validation_end_dict

    def configure_optimizers(self):
        """
        Return the optimizers.

        Returns:
            optimizer/s adopted.
        """
        return self.optimizers

    @pl.data_loader
    def tng_dataloader(self):
        """
        Training DataLoader.

        Returns:
            a torch.utils.data.DataLoader.
        """
        return self.data_loaders['train']

    @pl.data_loader
    def val_dataloader(self):
        """
        Validation DataLoader.

        Returns:
            a torch.utils.data.DataLoader.
        """
        return self.data_loaders['val']

    @pl.data_loader
    def test_dataloader(self):
        """
        Test DataLoader.

        Returns:
            a torch.utils.data.DataLoader. In case 'test' isn't a key
                of self.data_loaders, the validation one is returned.
        """
        return self.data_loaders.get('test', self.data_loaders['val'])