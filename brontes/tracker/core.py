"""Tracker core."""
import logging
logger = logging.getLogger('brontes::tracker::core')


class Tracker:

    def log_tensor_dict(self, dictionary, step):
        logger.debug(
            {key: value.item()
             for key, value in dictionary.items()}
        )
