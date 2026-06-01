# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
global_vars.py - Embodied training global state

Aligned with loongforge/utils/global_vars.py design:
  - set_model_config() / get_model_config(): model architecture cfg (OmegaConf, framework.*)
  - set_args() / get_args():                  training args (argparse Namespace)

Training params (lr, steps, gradient_checkpointing, freeze_vision_encoder, ...) live in args.
Architecture params (paligemma_variant, action_dim, ...) live in cfg.
"""

_EMBODIED_MODEL_CONFIG = None
_EMBODIED_ARGS = None


def set_model_config(cfg):
    """Store model architecture config globally. Must be called exactly once before training."""
    global _EMBODIED_MODEL_CONFIG
    assert _EMBODIED_MODEL_CONFIG is None, (
        "model config already set; set_model_config() should only be called once per process"
    )
    _EMBODIED_MODEL_CONFIG = cfg


def get_model_config():
    """Retrieve the globally stored model architecture config (OmegaConf)."""
    assert _EMBODIED_MODEL_CONFIG is not None, (
        "model config not initialized; call parse_train_args() first"
    )
    return _EMBODIED_MODEL_CONFIG


def set_args(args):
    """Store training args globally. Must be called exactly once before training."""
    global _EMBODIED_ARGS
    assert _EMBODIED_ARGS is None, (
        "args already set; set_args() should only be called once per process"
    )
    _EMBODIED_ARGS = args


def get_args():
    """Retrieve the globally stored training args (argparse Namespace)."""
    assert _EMBODIED_ARGS is not None, (
        "args not initialized; call parse_train_args() first"
    )
    return _EMBODIED_ARGS
