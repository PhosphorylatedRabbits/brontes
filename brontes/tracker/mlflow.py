"""Tracker module."""
import logging
import mlflow
from .core import Tracker

logger = logging.getLogger('brontes::tracker::mlflow')


class MLFlowTracker(Tracker):

    def log_tensor_dict(self, dictionary, step):

        mlflow.log_metrics(
            {key: value.item()
                for key, value in dictionary.items()},
            step=step
        )
