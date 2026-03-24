import logging
from functools import lru_cache
from pathlib import Path
from sys import stderr, stdout
import sys

FORMATTER1 = logging.Formatter('[{levelname}] - [{funcName}] - {message}', style='{')
FORMATTER2 = logging.Formatter('[{levelname}] - [{funcName}] - [{asctime}] - {message}', style='{', datefmt='%Y-%m-%d %H:%M:%S')

@lru_cache()
def get_logger():
    """
    Produce a logger instance. 

    The logger has two handlers:

    - one to `stdout` with level INFO

        Format includes function name.

    - one to `stderr` with level DEBUG
        
        Format includes function name and creation time.

    Notes
    -----
    This function is cached: the first call creates the instances, next calls read them from memory.

    Returns
    -------
    logging.Logger
        Ready-to-use logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stdout_ = logging.StreamHandler(stdout)
    stderr_ = logging.StreamHandler(stderr)

    stdout_.setLevel(logging.INFO)
    stderr_.setLevel(logging.DEBUG)

    stdout_.setFormatter(FORMATTER1)
    stderr_.setFormatter(FORMATTER2)

    logger.addHandler(stdout_)
    logger.addHandler(stderr_)

    return logger

def add_file_logging(log_file_path):
    logger = get_logger()
    # File handler for logger
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER1)
    logger.addHandler(file_handler)
    # Redirect stdout and stderr to the same file log_file_path
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
   
class StreamToLogger:
    """
    Redirects a stream (stdout/stderr) into the logger
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message, stacklevel=2)

    def flush(self):
        pass


def prepare_paths(*paths, as_str=False):
    new_paths = []
    for path in paths:
        p = Path(path).expanduser().resolve()
        if as_str:
            new_paths.append(str(p))
        else:
            new_paths.append(p)
    if len(new_paths) == 1:
        return new_paths[0]
    return new_paths