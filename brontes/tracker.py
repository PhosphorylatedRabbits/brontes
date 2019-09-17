"""Tracker module."""
import logging
from copy import deepcopy
logger = logging.getLogger('brontes::tracker')

# check if mlflow is available
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger.info(f'mlflow available: {MLFLOW_AVAILABLE}')


def _default_log_fn(dictionary, step):
    log_dictionary = deepcopy(dictionary)
    log_dictionary['step'] = step
    logger.debug(log_dictionary)


class Tracker:

    def __init__(self):
        if MLFLOW_AVAILABLE:
            self.log_fn = mlflow.log_metrics
        else:
            self.log_fn = _default_log_fn

    def log_tensor_dict(self, dictionary, step):

        self.log_fn(
            {key: value.item()
             for key, value in dictionary.items()},
            step=step
        )
