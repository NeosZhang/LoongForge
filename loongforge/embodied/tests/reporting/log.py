# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Console logger with colored output."""

import logging
import sys

_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[41m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if sys.stdout.isatty():
            color = _COLORS.get(record.levelno, "")
            return f"{color}{msg}{_RESET}"
        return msg


def create_logger(name="embodied_tests", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
