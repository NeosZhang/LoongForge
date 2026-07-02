# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument definitions — training control parameters (not model structure).

Parameter ownership:
  - arguments.py  → CLI training/data/running params  →  args (argparse Namespace)
  - YAML          → model architecture params          →  cfg (OmegaConf, via get_model_config())
"""

import argparse
from typing import Optional, Union


def _parse_reshard_after_forward(value: str) -> Optional[Union[bool, int]]:
    """Parse FSDP2 reshard_after_forward from CLI text."""
    normalized = value.strip().lower()
    if normalized in {"true", "t"}:
        return True
    if normalized in {"false", "f"}:
        return False
    if normalized in {"none", "null"}:
        return None
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected one of: true, false, none, or an integer greater than 1"
        ) from exc
    if int_value <= 1:
        raise argparse.ArgumentTypeError(
            "integer reshard_after_forward must be greater than 1"
        )
    return int_value


def _parse_reshard_after_forward_map(value: str) -> dict[str, Optional[Union[bool, int]]]:
    """Parse ClassName=value pairs for per-module FSDP reshard settings."""
    result = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "expected comma-separated ClassName=value pairs"
            )
        class_name, raw_value = item.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise argparse.ArgumentTypeError("empty class name in reshard map")
        result[class_name] = _parse_reshard_after_forward(raw_value)
    return result


def _parse_positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if int_value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int_value


def embodied_args_provider(parser: argparse.ArgumentParser):
    """Register model / training / distributed argument groups onto parser."""
    add_model_args(parser)
    add_training_args(parser)
    add_distributed_args(parser)
    add_cuda_graph_args(parser)


def add_model_args(parser: argparse.ArgumentParser):
    """Model configuration: model-name routing + training-time model switches."""
    g = parser.add_argument_group("Model Config")
    g.add_argument("--model-name", type=str, default=None,
                   help="Model name (maps to YAML via config_map)")
    g.add_argument("--config-file", type=str, default=None,
                   help="Direct path to YAML config (overrides --model-name)")
    g.add_argument("--tokenizer-path", type=str, default=None,
                   help="Path to tokenizer directory. Also settable via TOKENIZER_PATH env var.")
    g.add_argument("--trainer-type", type=str, required=True,
                   help="Trainer class name to use (e.g. FinetuneTrainer). "
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
    g.add_argument("--gradient-accumulation-steps", type=int, default=1)
    g.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing (memory <-> compute trade-off).")
    g.add_argument("--loss-spike-threshold", type=float, default=100.0)

    # ── Learning Rate ──
    g = parser.add_argument_group("Learning Rate")
    g.add_argument("--lr-base", type=float, default=2.5e-5,
                   help="Base learning rate for unmatched parameters.")
    g.add_argument("--lr-group", type=str, default=None,
                   help="Per-module LR overrides in 'module.path=lr' format, "
                        "comma-separated. "
                        "Example: 'model.paligemma_with_expert.gemma_expert=1e-4,"
                        "model.paligemma_with_expert=1e-5'. "
                        "Order matters: earlier entries consume parameters first.")
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
    g.add_argument("--dataloader-multiprocessing-context", type=str, default=None,
                   choices=["fork", "spawn", "forkserver"],
                   help="Multiprocessing start method for DataLoader workers. "
                        "Use spawn when video decoders are not fork-safe.")
    g.add_argument("--normalization-mode", type=str, default="q99")
    g.add_argument("--distributed-sampler-mode", type=str, default="cyclic",
                   choices=["cyclic", "block"],
                   help="How to partition the global index sequence across "
                        "data-parallel ranks (HPC-style naming). "
                        "'cyclic' — sample-level round-robin (a.k.a. stride sharding): "
                        "each rank gets indices[rank::world_size], a micro-batch is a "
                        "strided slice of the global order. This is what PyTorch's "
                        "DistributedSampler does. "
                        "'block' — batch-level round-robin (a.k.a. contiguous batch "
                        "sharding): the global order is first grouped into "
                        "size-batch_size blocks, then whole blocks are assigned to "
                        "ranks in round-robin fashion, so a micro-batch is a "
                        "contiguous slice. This matches HuggingFace Accelerate's "
                        "BatchSamplerShard(split_batches=False);")
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

    # ── Logging ──
    g = parser.add_argument_group("Logging")
    g.add_argument("--log-interval", type=int, default=1)
    g.add_argument("--detail-log-interval", type=int, default=20,
                   help="Interval (iters) for per-stage timing logs. 0 disables. "
                        "Aligned with AIAK-Training-Omni output format.")
    g.add_argument("--timing-log-level", type=int, default=0, choices=[0, 1],
                   help="Per-stage timing verbosity. 0: only max across ranks. "
                        "1: also print every rank's per-stage time.")
    g.add_argument("--loss-log-rank", nargs="+", type=int, default=[-1],
                   help="Loss logging mode. -1 (default): all-reduce (mean) the "
                        "loss across ranks so the reported value reflects the "
                        "global batch (logged on rank 0). One or more non-negative "
                        "ranks: skip the reduce and print the loss on each of those "
                        "ranks (tagged with its rank number), e.g. "
                        "`--loss-log-rank 0 3 7`.")
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
   

def add_cuda_graph_args(parser: argparse.ArgumentParser):
    """Add CUDA graph related CLI arguments."""

    # ── GR00TN1.6 CUDA Graph arguments ──
    g = parser.add_argument_group("CUDA Graph")
    g.add_argument("--cuda-graph-impl", type=str, default="none",
                   choices=["none", "local"],
                   help="CUDA graph implementation. Use 'local' to enable the embodied "
                        "trainer's local CUDA graph path.")
    g.add_argument("--cuda-graph-scope", type=str, default="full_iteration",
                   choices=["full_iteration", "per_microbatch"],
                   help="CUDA graph capture scope.")
    g.add_argument("--cuda-graph-warmup-steps", type=int, default=3,
                   help="Number of eager warmup iterations before CUDA graph capture.")
    g.add_argument("--cuda-graph-pad-length", type=int, default=None,
                   help="Pad token sequences to a fixed length for CUDA graph capture. "
                        "Use 0 to keep dynamic padding and let the graph runner recapture "
                        "when shapes change.")
    g.add_argument("--cuda-graph-ddp-sync-in-graph",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="When using local per-microbatch CUDA graph with DDP, wrap the "
                        "model in DDP and capture DDP gradient reductions into the graph "
                        "instead of replaying then synchronizing gradients manually.")
    g.add_argument("--cuda-graph-grad-sync-bucket-mb", type=float, default=200.0,
                   help="Bucket size in MiB for manual gradient all-reduce used by "
                        "per-microbatch CUDA graph when DDP reductions are not captured.")
    g.add_argument("--cuda-graph-grad-sync-impl", type=str, default="coalesced",
                   choices=["flat", "coalesced"],
                   help="Manual gradient all-reduce implementation for per-microbatch "
                        "CUDA graph. 'coalesced' avoids per-step flatten/unflatten copies.")
    g.add_argument("--cuda-graph-grad-sync-dtype", type=str, default="fp32",
                   choices=["fp32", "bf16"],
                   help="Communication dtype for manual CUDA graph gradient all-reduce. "
                        "Use bf16 to reduce communication volume; gradients are copied "
                        "back to their original dtype before optimizer/clip.")
    g.add_argument("--check-for-nan-in-loss-and-grad",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Enable host-side loss/grad NaN checks. CUDA graph mode disables "
                        "this automatically unless --no-check-for-nan-in-loss-and-grad is set.")


def add_distributed_args(parser: argparse.ArgumentParser):
    """Add all distributed training CLI arguments."""
    g = parser.add_argument_group("Distributed")
    g.add_argument("--distributed-strategy", type=str, default="fsdp",
                   choices=["ddp", "fsdp"])
    g.add_argument("--hsdp-shard-size", type=_parse_positive_int, default=None,
                   help="Enable HSDP and set the second 2D mesh dimension size. "
                        "The first mesh dimension replicates parameters across "
                        "groups, and this dimension shards parameters within "
                        "each group. Must divide the distributed world size. "
                        "Unset uses regular 1D FSDP.")
    g.add_argument("--fsdp-reshard-default", type=_parse_reshard_after_forward, default=None,
                   help="Default FSDP2 reshard_after_forward for non-root groups: "
                        "true, false, none, or an integer greater than 1. none uses "
                        "FSDP2 defaults.")
    g.add_argument("--fsdp-reshard-root", type=_parse_reshard_after_forward, default=False,
                   help="FSDP2 reshard_after_forward for the root group. Defaults to false.")
    g.add_argument("--fsdp-reshard-module-overrides",
                   type=_parse_reshard_after_forward_map, default=None,
                   help="Comma-separated ClassName=value overrides for FSDP "
                        "reshard_after_forward, e.g. GemmaMLP=false,Linear=true.")
    g.add_argument("--fsdp-wrap-modules", type=str, default=None,
                   help="Comma-separated class names for explicit FSDP wrap boundaries")
    g.add_argument("--fsdp-no-wrap-modules", type=str, default=None,
                   help="Comma-separated class names to decompose instead of wrapping directly")
    g.add_argument("--fsdp-min-num-params", type=int, default=1_000_000,
                   help="Minimum parameter count for automatic repeated-layer FSDP wrapping")
    g.add_argument("--fsdp-leftover-min-num-params", type=int, default=1_000_000,
                   help="Minimum parameter count for leftover dtype-based FSDP wrapping")
    g.add_argument("--ddp-broadcast-buffers", action=argparse.BooleanOptionalAction, default=True,
                   help="Broadcast module buffers from rank 0 at DDP forward start.")
    g.add_argument("--ddp-init-sync", action=argparse.BooleanOptionalAction, default=True,
                   help="Verify parameter shapes and broadcast params/buffers at DDP init.")
    g.add_argument("--ddp-bucket-cap-mb", type=int, default=None,
                   help="DDP gradient bucket size in MiB. None uses PyTorch default.")
    g.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=True,
                   help="Traverse autograd graph to detect unused DDP parameters.")
    g.add_argument("--ddp-gradient-as-bucket-view", action=argparse.BooleanOptionalAction, default=False,
                   help="Make gradients views into DDP buckets to reduce peak memory.")
    g.add_argument("--ddp-static-graph", action=argparse.BooleanOptionalAction, default=False,
                   help="Tell DDP that used/unused parameters and graph structure are static.")
    g.add_argument("--ddp-skip-all-reduce-unused-params", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip all-reduce for unused parameters when supported by the PyTorch version.")
    g.add_argument("--ddp-bucket-cap-mb-list", type=str, default=None,
                   help="Comma-separated DDP bucket sizes in MiB when supported by PyTorch.")
    g.add_argument("--ddp-batched-grad-copy", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable DDP batched gradient copy when supported by PyTorch.")
    g.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    g.add_argument("--zero-optimizer", action="store_true",
                   help="Wrap optimizer with ZeroRedundancyOptimizer (ZeRO Stage-1). "
                        "Shards optimizer states across ranks. Only effective with DDP.")
    g.add_argument("--zero-parameters-as-bucket-view", action=argparse.BooleanOptionalAction, default=False,
                   help="Pass parameters_as_bucket_view=True to ZeroRedundancyOptimizer. "
                        "Reduces peak memory by reusing gradient buffers as parameter storage, "
                        "but may conflict with torch.compile + DDP reducer assumptions. "
                        "Only effective when --zero-optimizer is set.")


def add_model_override_args(parser: argparse.ArgumentParser):
    """Allow CLI to override YAML model fields via dotlist positional args."""
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="YAML field overrides in dotlist format: backbone.image_size=448",
    )
