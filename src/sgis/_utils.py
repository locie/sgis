import logging
from functools import lru_cache
from pathlib import Path
from shutil import rmtree
from sys import stderr, stdout

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

    formatter1 = logging.Formatter('[{levelname}] - [{funcName}] - {message}', style='{')
    formatter2 = logging.Formatter('[{levelname}] - [{funcName}] - [{asctime}] - {message}', style='{', datefmt='%Y-%m-%d %H:%M:%S')

    stdout_ = logging.StreamHandler(stdout)
    stderr_ = logging.StreamHandler(stderr)

    stdout_.setLevel(logging.INFO)
    stderr_.setLevel(logging.DEBUG)

    stdout_.setFormatter(formatter1)
    stderr_.setFormatter(formatter2)

    logger.addHandler(stdout_)
    logger.addHandler(stderr_)

    return logger



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

def clean_processing_folder():
    """Delete the content of the Qgis processing temporary folder. Use with caution.
    """
    from processing import getTempFilename
    temp_filename = Path(getTempFilename())
    processing_folder = temp_filename.parent
    logger = get_logger()
    if processing_folder.is_dir():
        for path in processing_folder.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)
        logger.info(f"Content of folder '{processing_folder}' was deleted.")
