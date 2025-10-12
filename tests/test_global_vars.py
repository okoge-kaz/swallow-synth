import logging

import pytest

from src import global_vars


@pytest.fixture(autouse=True)
def reset_logger_state() -> None:
    global_vars._LOGGER = None  # type: ignore[attr-defined]


def test_init_logger_idempotent() -> None:
    logger1 = global_vars.init_logger(level=logging.DEBUG)
    logger2 = global_vars.init_logger(level=logging.ERROR)

    assert logger1 is logger2
    assert logger1.level == logging.DEBUG


def test_get_logger_initializes_when_missing() -> None:
    logger = global_vars.get_logger()
    assert logger is global_vars._LOGGER  # type: ignore[attr-defined]
