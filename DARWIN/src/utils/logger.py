"""Lightweight logger factory. Keeps logging consistent across modules."""

import logging
import sys

from src.utils import config


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    global _CONFIGURED
    if not _CONFIGURED:
        level = getattr(logging, config.DEFAULTS.get('log_level', 'INFO'))
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
            datefmt='%H:%M:%S',
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)