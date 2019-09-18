"""Tracker core."""
import logging

logger = logging.getLogger('brontes::tracker::core')


def default_log_tensor_dict(dictionary, step):
    """
    Log a dictionary containing tensors using the logging
    package at level debug.

    Args:
        dictionary (dict): a dictionary of torch.tensors.
        setp (int): integer step.
    """
    log_dictionary = {
        key: value.item()
        for key, value in dictionary.items()
    }
    log_dictionary['step'] = step
    logger.debug(log_dictionary)


class Tracker:
    """Tracker class."""

    def log_tensor_dict(self, dictionary, step):
        """
        Log a dictionary containing tensors using the logging
        package at level debug.

        Args:
            dictionary (dict): a dictionary of torch.tensors.
            setp (int): integer step.
        """
        default_log_tensor_dict(dictionary, step)
