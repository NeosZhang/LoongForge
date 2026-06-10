# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Misc training utilities."""

import contextlib
import logging
import os
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def is_rank_zero() -> bool:
    """Rank 0 check covering single-process, torchrun, and dist-initialized cases."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Fallback to env (torchrun sets RANK before init); single process defaults to 0.
    return int(os.environ.get("RANK", "0")) == 0


def set_seed(seed: int):
    """Set random seed across all sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool = True):
    """Enable or disable deterministic algorithms for reproducibility."""
    torch.use_deterministic_algorithms(enabled)


def setup_logging(output_dir: str, rank: int):
    """Configure logging: rank0 gets INFO + file handler, others get WARNING only.

    All handlers are registered in a single ``basicConfig(force=True)`` call
    so re-invocations (or prior implicit StreamHandlers) do not stack up and
    cause duplicate output.
    """
    level = logging.INFO if rank == 0 else logging.WARNING

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    handlers = [sh]

    if rank == 0 and output_dir:
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)


@contextlib.contextmanager
def log_stage(
    tag: str,
    start_msg: str = "",
    end_msg: str = "done in {elapsed}",
    log: Optional[logging.Logger] = None,
):
    """Context manager that wraps a setup stage with start/end logs + timing.

    Emits ``[tag] start_msg`` before entering the block and
    ``[tag] end_msg`` after leaving (the ``{elapsed}`` placeholder is
    interpolated with the formatted duration, e.g. ``"3.21s"``).

    Rank-0 gating is built in: when ``torch.distributed`` is initialized and
    the current rank is not 0, the context manager runs silently.

    Example
    -------
    >>> with log_stage("model", "building", "built in {elapsed}"):
    ...     model = build_model()
    """
    out = log or logger
    try:
        is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        is_main = True

    if is_main and start_msg:
        out.info(f"[{tag}] {start_msg}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if is_main and end_msg:
            elapsed = f"{time.perf_counter() - t0:.2f}s"
            out.info(f"[{tag}] {end_msg.format(elapsed=elapsed)}")


def _format_block(title: str, items: dict) -> str:
    """Render a Megatron-style key/value block.

    Layout:
        ------------------------ {title} ------------------------
          key1 ...................................... value
          key2 ...................................... value
        -------------------- end of {title} ---------------------
    """
    lines = [f"------------------------ {title} ------------------------"]
    for key in sorted(items.keys(), key=str.lower):
        dots = "." * max(1, 48 - len(key))
        lines.append(f"  {key} {dots} {items[key]}")
    lines.append(f"-------------------- end of {title} ---------------------")
    return "\n".join(lines)


def _flatten_omegaconf(cfg, prefix: str = "") -> dict:
    """Flatten OmegaConf into a dotted dict: {'a.b.c': value, ...}."""
    out = {}
    if isinstance(cfg, DictConfig):
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_omegaconf(v, key))
    elif isinstance(cfg, ListConfig):
        out[prefix] = OmegaConf.to_container(cfg, resolve=True)
    else:
        out[prefix] = cfg
    return out


def log_effective_config(args):
    """Log fully-resolved CLI args + model YAML config in Megatron-style format.

    Output is gated by the logger level: rank 0 is set to INFO and others to
    WARNING (see setup_logging / parser bootstrap), so non-rank-0 logger.info
    calls are filtered automatically — no explicit rank check needed here.
    """
    # CLI args (skip the attached OmegaConf to render it separately)
    args_items = {
        k: v for k, v in vars(args).items() if k != "model_cfg"
    }
    logger.info("\n%s", _format_block("arguments", args_items))

    # Flatten model YAML config to dotted keys
    cfg_items = _flatten_omegaconf(args.model_cfg)
    logger.info("\n%s", _format_block("model config", cfg_items))
