import logging
import sys
import typing

_LOGGER = None


def init_logger(level=logging.INFO) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("AppLogger")
    logger.setLevel(level)

    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _LOGGER = logger
    return _LOGGER


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        init_logger()
    return typing.cast(logging.Logger, _LOGGER)
