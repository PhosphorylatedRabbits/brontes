"""Tracker core."""
import logging
from copy import deepcopy
logger = logging.getLogger('brontes::tracker::core')


class Tracker:

    def log_tensor_dict(self, dictionary, step):
        log_dictionary = deepcopy(dictionary)
        log_dictionary['step'] = step
        logger.debug(
            {key: value.item()
             for key, value in dictionary.items()},
            step=step
        )
