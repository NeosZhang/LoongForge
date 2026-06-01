# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument definitions — training control parameters (not model structure).

Parameter ownership:
  - arguments.py  → CLI training/data/running params  →  args (argparse Namespace)
  - YAML          → model architecture params          →  cfg (OmegaConf, via get_model_config())
"""

import argparse


def add_model_args(parser: argparse.ArgumentParser):
    """Model configuration: model-name routing + training-time model switches."""
    g = parser.add_argument_group("Model Config")
    g.add_argument("--model-name", type=str, default=None,
                   help="Model name (maps to YAML via config_map)")
    g.add_argument("--training-phase", type=str, default="finetune",
                   choices=["pretrain", "finetune"])
    g.add_argument("--config-file", type=str, default=None,
                   help="Direct path to YAML config (overrides --model-name)")
    g.add_argument("--tokenizer-path", type=str, default=None,
                   help="Path to tokenizer directory. Also settable via TOKENIZER_PATH env var.")
    g.add_argument("--trainer-type", type=str, default=None,
                   help="Trainer class override (e.g. BCTrainer). If None, uses registry dispatch.")
    g.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing (memory <-> compute trade-off).")
    g.add_argument("--freeze-vision-encoder", action="store_true",
                   help="Freeze the vision tower (eval + requires_grad=False).")
    g.add_argument("--train-expert-only", action="store_true",
                   help="Train only the action expert; freeze the VLM backbone.")
    g.add_argument("--discrete-state-input", action="store_true",
                   help="Discretize robot state into 256 bins and embed in prompt (PI0.5 style).")


def add_training_args(parser: argparse.ArgumentParser):
    """Add all training-related CLI arguments."""

    # ── Basic Training ──
    g = parser.add_argument_group("Training")
    g.add_argument("--max-train-steps", type=int, default=150000)
    g.add_argument("--save-steps", type=int, default=10000)
    g.add_argument("--logging-frequency", type=int, default=50)
    g.add_argument("--seed", type=int, default=3047)
    g.add_argument("--output-dir", type=str, default="outputs/default")
    g.add_argument("--gradient-accumulation-steps", type=int, default=2)
    g.add_argument("--gradient-clipping", type=float, default=1.0)
    g.add_argument("--loss-spike-threshold", type=float, default=100.0)

    # ── Learning Rate ──
    g = parser.add_argument_group("Learning Rate")
    g.add_argument("--lr", type=float, default=2.5e-5, help="Base learning rate")
    g.add_argument("--lr-vlm", type=float, default=None, help="VLM interface LR override")
    g.add_argument("--lr-backbone", type=float, default=None, help="Backbone LR override (alias for --lr-vlm)")
    g.add_argument("--lr-action-model", type=float, default=None, help="Action model LR override")
    g.add_argument("--lr-scheduler-type", type=str, default="cosine_with_min_lr")
    g.add_argument("--warmup-steps", type=int, default=2000)
    g.add_argument("--min-lr", type=float, default=1e-6)

    # ── Optimizer ──
    g = parser.add_argument_group("Optimizer")
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.95)
    g.add_argument("--adam-eps", type=float, default=1e-8)

    # ── Distributed ──
    g = parser.add_argument_group("Distributed")
    g.add_argument("--distributed-strategy", type=str, default="fsdp",
                   choices=["ddp", "fsdp"])
    g.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    g.add_argument("--fsdp-wrap-modules", type=str, default=None,
                   help="Comma-separated class names for FSDP auto-wrap")
    g.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # ── Data ──
    g = parser.add_argument_group("Data")
    g.add_argument("--dataloader-module", type=str, default="lerobot_datasets")
    g.add_argument("--dataset-path", type=str, default=None)
    g.add_argument("--dataset-mix", type=str, default=None,
                   help="Named dataset mixture (e.g., libero_spatial, oxe_magic_soup)")
    g.add_argument("--data-root-dir", type=str, default="/data/lerobot")
    g.add_argument("--robot-type", type=str, default="libero_franka")
    g.add_argument("--task-name", type=str, default="perform the task",
                   help="Language task description (for HDF5 datasets)")
    g.add_argument("--per-device-batch-size", type=int, default=4)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--action-horizon", type=int, default=10)
    g.add_argument("--action-dim", type=int, default=7)
    g.add_argument("--state-dim", type=int, default=7)
    g.add_argument("--image-size", type=int, default=224)
    g.add_argument("--normalization-mode", type=str, default="q99")
    g.add_argument("--num-samples", type=int, default=100,
                   help="Number of samples for dummy dataset")

    # ── Checkpoint ──
    g = parser.add_argument_group("Checkpoint")
    g.add_argument("--pretrained-checkpoint", type=str, default=None)
    g.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    g.add_argument("--save-format", type=str, default="safetensors",
                   choices=["safetensors", "pt"])
    g.add_argument("--save-training-state", action="store_true")

    # ── Freeze ──
    g = parser.add_argument_group("Freeze")
    g.add_argument("--freeze-modules", type=str, default="",
                   help="Comma-separated module paths to freeze")

    # ── EMA ──
    g = parser.add_argument_group("EMA")
    g.add_argument("--ema", action="store_true")
    g.add_argument("--ema-decay", type=float, default=0.9999)

    # ── LoRA ──
    g = parser.add_argument_group("LoRA")
    g.add_argument("--lora", action="store_true")
    g.add_argument("--lora-rank", type=int, default=8)
    g.add_argument("--lora-alpha", type=int, default=16)
    g.add_argument("--lora-target-modules", type=str, default="all-linear")

    # ── Logging ──
    g = parser.add_argument_group("W&B")
    g.add_argument("--wandb-project", type=str, default="loongforge-vla")
    g.add_argument("--wandb-mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])


def add_model_override_args(parser: argparse.ArgumentParser):
    """Allow CLI to override YAML model fields via dotlist positional args."""
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="YAML field overrides in dotlist format: backbone.image_size=448",
    )
