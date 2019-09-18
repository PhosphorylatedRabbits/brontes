"""Tracker factory."""
import logging

from .core import Tracker

logger = logging.getLogger('brontes::tracker::init')
# instantiate the default tracker
TRACKER_FACTORY = {
    'logging': Tracker
}
# add mlflow support if available
try:
    from .mlflow import MLflowTracker
    TRACKER_FACTORY['mlflow'] = MLflowTracker
except ImportError:
    logger.warning('mlflow not found: MLflowTracker not available')

logger.info(f'available trackers: {TRACKER_FACTORY.keys()}')
