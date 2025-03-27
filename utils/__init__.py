from . import input_utils, logger_utils, quote_utils, save_utils, string_utils
from .input_utils import InputType
from .logger_utils import LoggingCallback
from .timer_utils import timer


def get_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
