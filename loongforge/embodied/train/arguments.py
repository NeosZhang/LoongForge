# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument definitions — training control parameters (not model structure).

Parameter ownership:
  - arguments.py  → CLI training/data/running params  →  args (argparse Namespace)
  - YAML          → model architecture params          →  cfg (OmegaConf, via get_model_config())
"""

import argparse

def embodied_args_provider(parser: argparse.ArgumentParser):
    """Register model / training / distributed argument groups onto parser."""
    add_model_args(parser)
    add_training_args(parser)
    add_distributed_args(parser)


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
    g.add_argument("--trainer-type", type=str, required=True,
                   help="Trainer class name to use (e.g. BCTrainer). "
                        "See _TRAINER_CLASSES in trainer_builder.py for supported values.")
    g.add_argument("--freeze-vision-encoder", action="store_true",
                   help="Freeze the vision tower (eval + requires_grad=False).")
    g.add_argument("--train-expert-only", action="store_true",
                   help="Train only the action expert; freeze the VLM backbone.")


def add_training_args(parser: argparse.ArgumentParser):
    """Add all training-related CLI arguments."""

    # ── Basic Training ──
    g = parser.add_argument_group("Training")
    g.add_argument("--train-iters", type=int, default=150000)
    g.add_argument("--save-interval", type=int, default=10000)
    g.add_argument("--seed", type=int, default=3047)
    g.add_argument("--deterministic-mode", action="store_true",
                   help="Force cuDNN deterministic mode for reproducibility (may slow training).")
    g.add_argument("--output-dir", type=str, default="outputs/default",
                   help="Root output directory for checkpoints, logs and run artifacts.")
    g.add_argument("--gradient-accumulation-steps", type=int, default=2)
    g.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing (memory <-> compute trade-off).")
    g.add_argument("--loss-spike-threshold", type=float, default=100.0)

    # ── Learning Rate ──
    g = parser.add_argument_group("Learning Rate")
    g.add_argument("--lr", type=float, default=2.5e-5, help="Base learning rate")
    g.add_argument("--lr-backbone", type=float, default=None, help="Backbone LR override")
    g.add_argument("--lr-action-model", type=float, default=None, help="Action model LR override")
    g.add_argument("--lr-decay-style", type=str, default="cosine_with_min_lr")
    g.add_argument("--lr-warmup-iters", type=int, default=2000)
    g.add_argument("--min-lr", type=float, default=1e-6)

    # ── Optimizer ──
    g = parser.add_argument_group("Optimizer")
    g.add_argument("--optimizer", type=str, default="AdamW")
    g.add_argument("--clip-grad", type=float, default=1.0,
                   help="Gradient clipping norm.")
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.95)
    g.add_argument("--adam-eps", type=float, default=1e-8)

    # ── Data ──
    g = parser.add_argument_group("Data")
    g.add_argument("--dataloader-module", type=str, default="lerobot_datasets")
    g.add_argument("--dataset-path", type=str, default=None)
    g.add_argument("--lerobotdataset-version", type=str, default="v3.0",
                   choices=["v2.0", "v2.1", "v3.0"],
                   help="LeRobot dataset format version. v2.0/v2.1 uses JSONL + parquet "
                        "(no lerobot lib needed); v3.0 uses official LeRobotDataset API.")
    g.add_argument("--video-backend", type=str, default="torchcodec",
                   choices=["torchcodec", "decord", "opencv", "pyav", "torchvision_av"],
                   help="Video decoding backend for dataset loading.")
    g.add_argument("--streaming", action="store_true",
                   help="Use streaming (iterable) dataset mode for v3.0 format.")
    g.add_argument("--dataset-mix", type=str, default=None,
                   help="Name of a registered dataset mixture to load; expands to one or more datasets "
                        "under --data-root-dir (e.g., libero_spatial, oxe_magic_soup). Mutually "
                        "exclusive with --dataset-path.")
    g.add_argument("--data-root-dir", type=str, default=None,
                   help="Root directory used to resolve datasets in --dataset-mix; each registered "
                        "dataset path is joined with this directory.")
    g.add_argument("--robot-type", type=str, default=None,
                   help="Robot embodiment type (e.g. libero_franka, aloha, ur5).")
    g.add_argument("--task-name", type=str, default="perform the task",
                   help="Language task description (for HDF5 datasets)")
    g.add_argument("--per-device-batch-size", type=int, default=4)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--normalization-mode", type=str, default="q99")
    g.add_argument("--discrete-state-input", action="store_true",
                   help="Discretize robot state into 256 bins and embed in prompt (PI0.5 style).")
    g.add_argument("--num-samples", type=int, default=100,
                   help="Number of samples for dummy dataset")

    # ── Checkpoint ──
    g = parser.add_argument_group("Checkpoint")
    g.add_argument("--pretrained-checkpoint", type=str, default=None)
    g.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    g.add_argument("--save-format", type=str, default="safetensors",
                   choices=["safetensors", "pt", "dcp"],
                   help="Checkpoint format: 'safetensors'/'pt' = rank0 consolidated "
                        "single file (legacy); 'dcp' = per-rank sharded via "
                        "torch.distributed.checkpoint (low memory, supports "
                        "world-size resharding on resume).")
    g.add_argument("--save-training-state", action=argparse.BooleanOptionalAction, default=True,
                   help="Save optimizer/scheduler/RNG state alongside model weights "
                        "(required for true resume). Use --no-save-training-state for "
                        "weights-only export.")
    g.add_argument("--async-save", action="store_true",
                   help="Use torch.distributed.checkpoint.async_save to overlap "
                        "checkpoint I/O with training. Only effective when "
                        "--save-format=dcp (PyTorch >= 2.4).")

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
    g = parser.add_argument_group("Logging")
    g.add_argument("--log-interval", type=int, default=1)
    g.add_argument("--detail-log-interval", type=int, default=20,
                   help="Interval (iters) for per-stage timing logs. 0 disables. "
                        "Aligned with AIAK-Training-Omni output format.")
    g.add_argument("--timing-log-level", type=int, default=0, choices=[0, 1],
                   help="Per-stage timing verbosity. 0: only max across ranks. "
                        "1: also print every rank's per-stage time.")
    g.add_argument("--wandb-project", type=str, default="loongforge-vla")
    g.add_argument("--wandb-mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    g.add_argument("--tensorboard-dir", type=str, default=None,
                   help="Directory to write TensorBoard event files. When unset, "
                        "TensorBoard logging is disabled. When set to a relative path, "
                        "it is resolved against --output-dir.")
    g.add_argument("--tensorboard-queue-size", type=int, default=1000,
                   help="Size of the queue for asynchronous TensorBoard event writes.")

    # ── Profiler ──
    # torch.profiler integration. Produces TensorBoard-compatible traces
    # via tensorboard_trace_handler.
    g = parser.add_argument_group("Profiler")
    g.add_argument("--use-pytorch-profiler", action="store_true",
                   dest="use_pytorch_profiler",
                   help="Enable torch.profiler. Writes traces via "
                        "tensorboard_trace_handler to --profile-output-dir.")
    g.add_argument("--use-nsys-profiler", action="store_true",
                   dest="use_nsys_profiler",
                   help="Enable nsys-style profiling: cudaProfilerStart/Stop "
                        "+ autograd emit_nvtx ranges, scoped to "
                        "[profile_step_start, profile_step_end). Run the "
                        "training command under `nsys profile -c cudaProfilerApi ...` "
                        "to capture only the marked region. "
                        "Mutually exclusive with --use-pytorch-profiler.")
    g.add_argument("--profile-step-start", type=int, default=10,
                   help="Global step at which to start profiling.")
    g.add_argument("--profile-step-end", type=int, default=12,
                   help="Global step at which to stop profiling.")
    g.add_argument("--profile-ranks", nargs="+", type=int, default=[0],
                   help="Global ranks to profile (applies to both pytorch and nsys).")
    g.add_argument("--profile-output-dir", type=str, default=None,
                   help="Directory to write torch.profiler traces. Defaults "
                        "to {output_dir}/profiler.")
    

def add_distributed_args(parser: argparse.ArgumentParser):
    """Add all distributed training CLI arguments."""
    g = parser.add_argument_group("Distributed")
    g.add_argument("--distributed-strategy", type=str, default="fsdp",
                   choices=["ddp", "fsdp"])
    g.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    g.add_argument("--fsdp-wrap-modules", type=str, default=None,
                   help="Comma-separated class names for FSDP auto-wrap")
    g.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    g.add_argument("--zero-optimizer", action="store_true",
                   help="Wrap optimizer with ZeroRedundancyOptimizer (ZeRO Stage-1). "
                        "Shards optimizer states across ranks. Only effective with DDP.")


def add_model_override_args(parser: argparse.ArgumentParser):
    """Allow CLI to override YAML model fields via dotlist positional args."""
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="YAML field overrides in dotlist format: backbone.image_size=448",
    )
