# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Unified argument parsing: YAML (model structure) + CLI (training control).

Design: args and cfg are strictly separated:
  - args (argparse Namespace): all training/data/running params, used directly by trainer
  - cfg  (OmegaConf):          model architecture params only (framework.*), stored globally

Usage:
    from embodied.train.parser import parse_train_args
    args = parse_train_args()
    # cfg available via: from embodied.train.global_vars import get_model_config
"""

import argparse
import logging
import os

from omegaconf import OmegaConf

from .arguments import add_model_override_args, embodied_args_provider
from .config_map import get_config_path
from .global_vars import set_args, set_model_config
from .validators import validate_args

logger = logging.getLogger(__name__)


def parse_train_args():
    """
    Parse flow:
      1. Top-level CLI: --model-name / --training-phase / --config-file + model switches
      2. Parse all CLI training args → args namespace
      3. Load YAML → model_cfg (OmegaConf, model structure only)
      4. Process positional overrides → override model_cfg fields
      5. Store cfg globally via set_model_config(), args via set_args()
      6. Validate args + cfg
      7. Attach model_cfg to args.model_cfg (for backward compat)
      8. Return args; cfg also accessible via get_model_config()
    """
    parser = argparse.ArgumentParser(
        description="LoongForge Embodied Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model routing + switches + training + distributed
    embodied_args_provider(parser)

    # Model field overrides (positional dotlist)
    add_model_override_args(parser)

    args = parser.parse_args()

    # Propagate tokenizer path to env var (model builder reads TOKENIZER_PATH)
    if args.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = args.tokenizer_path

    # ── Load model structure YAML ──
    if args.config_file:
        config_path = args.config_file
    elif args.model_name:
        config_path = get_config_path(args.model_name)
    else:
        raise ValueError("Must specify --model-name or --config-file")

    model_cfg = OmegaConf.load(config_path)

    # Apply CLI dotlist overrides to model config
    if args.overrides:
        override_cfg = OmegaConf.from_dotlist(args.overrides)
        model_cfg = OmegaConf.merge(model_cfg, override_cfg)

    # Store globally (aligned with loongforge main-path pattern)
    set_model_config(model_cfg)
    set_args(args)

    # Validate
    validate_args(args, model_cfg)

    # Backward compat: attach to args for trainer access
    args.model_cfg = model_cfg
    args.config_path = config_path

    logger.info(f"Model config loaded: {config_path}")
    logger.info(f"model_type={model_cfg.get('model_type', 'unknown')}")
    logger.info(f"Training: train_iters={args.train_iters}, lr={args.lr}, "
                f"batch_size={args.per_device_batch_size}")

    return args
