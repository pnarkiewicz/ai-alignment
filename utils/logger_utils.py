import logging
import os

from transformers import TrainerCallback


def get_log_level():
    """Gets the log level specified in the environment variables"""
    if "LOG_LEVEL" in os.environ:
        requested = os.environ["LOG_LEVEL"]
        for level in filter(lambda x: requested == str(x), [logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR]):
            return level
        return logging.INFO


def get_default_logger(name: str, log_level=None):
    """Generates a logger at the specified log level and in the specified namespace"""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        log_level = log_level or get_log_level()

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)

        logger.setLevel(log_level)
        logger.addHandler(stream_handler)
    return logger


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Callback so that training updates get written using the default logger rather than print statements"""
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            get_default_logger(__name__).warn(logs)
