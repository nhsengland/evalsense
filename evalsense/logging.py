import json
import logging
from logging import Logger
from logging.config import dictConfig
import os
from pathlib import Path

EVALSENSE_LOGGING_PREFIX = os.environ.get("EVALSENSE_LOGGING_PREFIX", "")
EVALSENSE_LOGGING_LEVEL = os.environ.get("EVALSENSE_LOGGING_LEVEL", "INFO").upper()
EVALSENSE_LOGGING_CONFIG_PATH = os.environ.get("EVALSENSE_LOGGING_CONFIG_PATH", None)

_FORMAT = (
    f"{EVALSENSE_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(filename)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "default": {
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": EVALSENSE_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "evalsense": {
            "level": EVALSENSE_LOGGING_LEVEL,
            "handlers": ["console"],
            "propagate": False,
        }
    },
    "version": 1,
    "disable_existing_loggers": False,
}


def get_logger(name: str) -> Logger:
    """Sets up logging for the application."""
    if EVALSENSE_LOGGING_CONFIG_PATH:
        config_path = Path(EVALSENSE_LOGGING_CONFIG_PATH)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Specified logging config file {EVALSENSE_LOGGING_CONFIG_PATH} "
                "does not exist."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = DEFAULT_LOGGING_CONFIG

    dictConfig(config)

    return logging.getLogger(name)
