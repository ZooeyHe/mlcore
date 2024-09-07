"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-07-04
"""

import logging
import os
import sys

from .formatters import ConsoleColorFormatter, FileDecoloringFormatter

"""
Implementation of custom colored logging.
"""


DEFAULT_FMT = "[%(asctime)s][%(module)8.8s:%(lineno)3d] %(levelname)7.7s: %(message)s"
DEFAULT_DATEFMT = "%Y%m%d%H%M%S"


def setup_console_logging(
    app_name: str,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATEFMT,
    level: int = logging.WARN
) -> logging.Logger:
    """
    Given an app name, set up the console logger (with color).
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleColorFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)
    return logger


def setup_file_logging(
    app_name: str,
    filename: str,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATEFMT,
    level: int = logging.WARN
) -> logging.Logger:
    """
    Given an app name, set up the logger fo writing to file.
    """
    dirname = os.path.dirname(os.path.abspath(filename))
    os.makedirs(dirname, exist_ok=True)  # Create the directory to save the log file.

    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    # Create and add handler to logger.
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(FileDecoloringFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)
    return logger
