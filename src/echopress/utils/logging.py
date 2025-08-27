"""Minimal logging helpers for the project."""

from __future__ import annotations

import logging
from typing import Optional

DEFAULT_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def get_logger(name: str = "echopress", level: int = logging.INFO, fmt: str = DEFAULT_FORMAT) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    A new ``StreamHandler`` is added only once per-logger to avoid
    duplicate log lines when calling this function multiple times.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
