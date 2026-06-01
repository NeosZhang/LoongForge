# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied Training Entry."""

import sys
import os

# Ensure embodied can be imported independently without triggering
# the main loongforge package (which requires Megatron).
# Add 'loongforge/' to sys.path so 'embodied.*' resolves directly.
_EMBODIED_DIR = os.path.dirname(os.path.abspath(__file__))
_LOONGFORGE_DIR = os.path.dirname(_EMBODIED_DIR)
if _LOONGFORGE_DIR not in sys.path:
    sys.path.insert(0, _LOONGFORGE_DIR)

from loongforge.embodied.train.parser import parse_train_args
from loongforge.embodied.train.trainers import build_model_trainer


def main():
    """Parse args, build trainer, and start training loop."""
    args = parse_train_args()
    trainer = build_model_trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
