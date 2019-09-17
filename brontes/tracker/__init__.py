"""Tracker factory."""
import logging
from .core import Tracker

logger = logging.getLogger('brontes::tracker::mlflow')
TRACKER_FACTORY = {
    'logging': Tracker
}

try:
    from .mlflow import MLFlowTracker
    TRACKER_FACTORY['mlflow'] = MLFlowTracker
except ImportError:
    logger.warning('mlflow not found')

logger.info(f'available trackers: {TRACKER_FACTORY.keys()}')
