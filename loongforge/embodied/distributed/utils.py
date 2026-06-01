# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""General training utilities."""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed across all sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str, rank: int):
    """Configure logging: rank0 gets INFO + file handler, others get WARNING only."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0 and output_dir:
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(fh)
