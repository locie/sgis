import logging
from functools import lru_cache
from pathlib import Path
from shutil import rmtree

@lru_cache()
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # fixme
    formatter = logging.Formatter('[{levelname}] - [{funcName}] - {message}', style='{')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
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
