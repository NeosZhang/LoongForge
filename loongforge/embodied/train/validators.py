# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
validators.py - VLA training argument validation

Called by parse_train_args() after args and cfg are loaded.
"""

import logging
import os

logger = logging.getLogger(__name__)


def validate_args(args, cfg):
    """
    Validate the combination of CLI args and architecture cfg.

    Raises ValueError on hard errors, logs warnings for soft issues.
    """
    # Model config must be specified
    if args.model_name is None and args.config_file is None:
        raise ValueError("--model-name or --config-file must be specified.")

    # LR sanity checks
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if args.min_lr < 0:
        raise ValueError(f"--min-lr must be >= 0, got {args.min_lr}")
    if args.min_lr >= args.lr:
        logger.warning(
            f"--min-lr ({args.min_lr}) >= --lr ({args.lr}); "
            f"cosine decay will have no effect."
        )

    # Steps
    if args.max_train_steps <= 0:
        raise ValueError(f"--max-train-steps must be positive, got {args.max_train_steps}")
    if args.warmup_steps >= args.max_train_steps:
        logger.warning(
            f"--warmup-steps ({args.warmup_steps}) >= --max-train-steps ({args.max_train_steps})"
        )

    # Checkpoint
    if args.resume and not args.pretrained_checkpoint:
        # Resume without explicit checkpoint is fine (will auto-find in output_dir)
        pass

    # Tokenizer
    if args.tokenizer_path is None:
        if not os.environ.get("TOKENIZER_PATH"):
            logger.warning(
                "Neither --tokenizer-path nor TOKENIZER_PATH env var is set. "
                "Model initialization may fail if a tokenizer is required."
            )

    # Architecture cfg structure
    if not hasattr(cfg, "model_type") and not hasattr(cfg, "framework"):
        logger.warning(
            f"Model config YAML has no 'model_type' or 'framework' top-level key. "
            f"Got keys: {list(cfg.keys())}"
        )
