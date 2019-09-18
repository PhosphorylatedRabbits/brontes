"""Tracker module using MLflow."""
import logging
import mlflow

from .core import Tracker

logger = logging.getLogger('brontes::tracker::mlflow')


class MLflowTracker(Tracker):
    """Tracker using MLLflow."""

    def log_tensor_dict(self, dictionary, step):
        """
        Log a dictionary containing tensors MLflow

        Args:
            dictionary (dict): a dictionary of torch.tensors.
            setp (int): integer step.
        """
        mlflow.log_metrics(
            {key: value.item()
                for key, value in dictionary.items()},
            step=step
        )
