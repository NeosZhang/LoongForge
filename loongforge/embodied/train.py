# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied Training Entry."""

from loongforge.embodied.train.parser import parse_train_args
from loongforge.embodied.train.trainers import build_model_trainer


def main():
    """Parse args, build trainer, and start training loop."""
    args = parse_train_args()
    trainer = build_model_trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
