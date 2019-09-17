import logging
logger = logging.getLogger('brontes::tracker')

# check if mlflow is available
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger.info(f'mlflow available: {MLFLOW_AVAILABLE}')


class Tracker:
    def __init__(self):
        if MLFLOW_AVAILABLE:
            self.log_fn = mlflow.log_metrics
        else:
            self.log_fn = logger.debug

    def log_tensor_dict(self, dictionary):

        self.log_fn({key: value.item() for key, value in dictionary.items()})
