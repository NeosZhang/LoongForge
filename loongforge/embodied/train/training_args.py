# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generic training args + CLI front-end (single module).

This module is the single source of truth for the generic, model-independent
training parameters (``TrainingArgs``, a frozen dataclass instantiated from the
CLI) plus the tooling that turns that dataclass into an argparse CLI.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``training_args.lr_base``.
2. Never use ``getattr(training_args, "x", default)`` or ``cfg.get("x", default)``:
   - a default supplied there creates a second source of truth and hides the real one;
   - a misspelled field should raise ``AttributeError`` immediately, not silently return
     a fallback.
3. To add or change a generic parameter, edit only this dataclass
   (one authoritative definition); the CLI flags, ``--help``, and the parameter
   summary are all generated from this dataclass by reflection.

Boundary: model-structure switches (freeze_vision_encoder, train_expert_only,
gradient_checkpointing, compile_model, ...) live in the per-model ModelConfig
(YAML model:); data-processing params (image_size, normalization_mode, ...) live
in the per-model DataConfig (YAML data:). They are intentionally NOT here.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional, get_args, get_origin, Union


# ═══════════════════════════════════════════════════════════════════
# Custom CLI value parsers (referenced by field metadata below)
# ═══════════════════════════════════════════════════════════════════


def parse_reshard_after_forward(value: str):
    """Parse FSDP2 reshard_after_forward from CLI text: true|false|none|int>1."""
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


def parse_reshard_after_forward_map(value: str):
    """Parse comma-separated ClassName=value pairs for per-module FSDP reshard."""
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
        result[class_name] = parse_reshard_after_forward(raw_value)
    return result


def parse_positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if int_value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int_value


# ═══════════════════════════════════════════════════════════════════
# TrainingArgs — single source of truth for generic training params
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrainingArgs:
    """Generic training args (single source of truth). Frozen after construction."""

    # ── Model routing (which YAML / trainer / tokenizer) ──
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model identifier (e.g. 'pi05', 'groot_n1_6'). Selects the "
                          "ModelConfig/DataConfig classes and default YAML via MODEL_SCHEMA. Required."})
    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit path to a model YAML config. Overrides the default YAML "
                          "resolved from --model-name; --model-name is still required to pick the config classes."})
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Directory or HF repo id of the tokenizer. Exported to the TOKENIZER_PATH "
                          "env var so the model/data tokenizer loaders can pick it up."})
    trainer_type: str = field(
        default="FinetuneTrainer",
        metadata={"help": "Trainer class to instantiate (e.g. FinetuneTrainer, PretrainTrainer); "
                          "resolved by the trainer builder registry."})

    # ── Basic Training ──
    train_iters: int = field(
        default=150000,
        metadata={"help": "Total number of optimizer update steps to run before stopping."})
    save_interval: int = field(
        default=10000,
        metadata={"help": "Write a checkpoint every N iterations."})
    seed: int = field(
        default=3047,
        metadata={"help": "Global RNG seed for Python/NumPy/PyTorch and data shuffling (reproducibility)."})
    deterministic_mode: bool = field(
        default=False,
        metadata={"help": "Force cuDNN deterministic algorithms. Improves reproducibility at some "
                          "throughput cost; requires CUBLAS_WORKSPACE_CONFIG to be set."})
    output_dir: str = field(
        default="outputs/default",
        metadata={"help": "Root directory for checkpoints, logs, and other run artifacts."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of micro-batches accumulated before each optimizer step; "
                          "effective batch = per_device_batch_size * world_size * this value."})
    loss_spike_threshold: float = field(
        default=100.0,
        metadata={"help": "If loss exceeds this value the step is treated as a spike (guard / skip)."})

    # ── Learning Rate ──
    lr_base: float = field(
        default=2.5e-5,
        metadata={"help": "Base learning rate applied to all parameters not matched by --lr-group."})
    lr_group: Optional[str] = field(
        default=None,
        metadata={"help": "Per-module LR overrides as 'module.path=lr,module2.path=lr2'. "
                          "Matched modules use the given LR instead of --lr-base."})
    lr_decay_style: str = field(
        default="cosine_with_min_lr",
        metadata={"help": "LR schedule shape (e.g. constant, linear, cosine, cosine_with_min_lr)."})
    lr_warmup_iters: int = field(
        default=2000,
        metadata={"help": "Number of iterations to linearly warm up the LR from 0 to its peak."})
    min_lr: float = field(
        default=1e-6,
        metadata={"help": "Lower bound the LR schedule decays to (floor)."})

    # ── Optimizer ──
    optimizer: str = field(
        default="AdamW",
        metadata={"help": "Optimizer name (e.g. AdamW, SGD)."})
    clip_grad: float = field(
        default=1.0,
        metadata={"help": "Max global gradient norm for clipping; <=0 disables clipping."})
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Decoupled weight decay coefficient (AdamW)."})
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta1 — exponential decay rate for the first moment (mean)."})
    adam_beta2: float = field(
        default=0.95,
        metadata={"help": "Adam beta2 — exponential decay rate for the second moment (variance)."})
    adam_eps: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon added to the denominator for numerical stability."})

    # ── Data loading control (cross-model; per-model processing lives in DataConfig) ──
    dataloader_module: str = field(
        default="lerobot_datasets",
        metadata={"help": "Dataset builder module to use (e.g. lerobot_datasets, rlds, hdf5, dummy)."})
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Filesystem path or repo id of the dataset to train on."})
    dataset_name: Optional[str] = field(
        default="bridge_v2",
        metadata={"help": "RLDS dataset name / Open-X-Embodiment key (RLDS loader only)."})
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to load (RLDS), e.g. 'train' or 'train[:95%]'."})
    lerobotdataset_version: str = field(
        default="v3.0",
        metadata={"choices": ["v2.0", "v2.1", "v3.0"],
                  "help": "On-disk LeRobot dataset format version to parse."})
    video_backend: str = field(
        default="torchcodec",
        metadata={"choices": ["torchcodec", "decord", "opencv", "pyav", "torchvision_av"],
                  "help": "Backend used to decode episode videos into frames."})
    streaming: bool = field(
        default=False,
        metadata={"help": "Use a streaming/iterable dataset instead of map-style random access "
                          "(lower memory, no global shuffle)."})
    dataset_mix: Optional[str] = field(
        default=None,
        metadata={"help": "Name of a registered dataset mixture (weighted multi-dataset sampling)."})
    data_root_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory containing the datasets referenced by --dataset-mix."})
    robot_type: Optional[str] = field(
        default=None,
        metadata={"help": "Robot embodiment type (e.g. libero_franka); selects action/state layout."})
    task_name: str = field(
        default="perform the task",
        metadata={"help": "Language instruction used as the prompt when the dataset has none (HDF5)."})
    per_device_batch_size: int = field(
        default=4,
        metadata={"help": "Micro-batch size processed per GPU per forward pass."})
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of DataLoader worker processes per rank."})
    dataloader_multiprocessing_context: Optional[str] = field(
        default=None,
        metadata={"choices": ["fork", "spawn", "forkserver"],
                  "help": "Multiprocessing start method for DataLoader workers."})
    distributed_sampler_mode: str = field(
        default="cyclic",
        metadata={"choices": ["cyclic", "block"],
                  "help": "How the distributed sampler partitions indices across ranks: "
                          "'cyclic' (round-robin) or 'block' (contiguous shards)."})
    discrete_state_input: bool = field(
        default=False,
        metadata={"help": "Discretize the robot state and inject it into the text prompt (PI0.5 style)."})
    num_samples: int = field(
        default=100,
        metadata={"help": "Number of synthetic samples to generate for the dummy dataset."})

    # ── Checkpoint ──
    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained weights to initialize the model from (fine-tuning)."})
    resume: bool = field(
        default=False,
        metadata={"help": "Resume training (weights + optimizer/scheduler/RNG state) from the "
                          "latest checkpoint in --output-dir."})
    save_format: str = field(
        default="safetensors",
        metadata={"choices": ["safetensors", "pt", "dcp"],
                  "help": "On-disk checkpoint format: safetensors, raw torch .pt, or distributed "
                          "checkpoint (dcp)."})
    save_training_state: bool = field(
        default=True,
        metadata={"help": "Also save optimizer, LR scheduler, and RNG state (needed to resume)."})
    async_save: bool = field(
        default=False,
        metadata={"help": "Save checkpoints asynchronously in the background (dcp format only)."})

    # ── Freeze ──
    freeze_modules: str = field(
        default="",
        metadata={"help": "Comma-separated module path prefixes whose parameters are frozen "
                          "(requires_grad=False)."})

    # ── Logging ──
    log_interval: int = field(
        default=1,
        metadata={"help": "Log scalar metrics (loss, LR, throughput) every N iterations."})
    detail_log_interval: int = field(
        default=20,
        metadata={"help": "Log detailed per-stage timing breakdown every N iterations."})
    timing_log_level: int = field(
        default=0,
        metadata={"choices": [0, 1],
                  "help": "Verbosity of per-stage timing logs: 0 = summary, 1 = detailed."})
    loss_log_rank: List[int] = field(
        default_factory=lambda: [-1],
        metadata={"help": "Ranks whose loss is logged; -1 logs the all-reduced mean across ranks."})
    wandb_project: str = field(
        default="loongforge-vla",
        metadata={"help": "Weights & Biases project name."})
    wandb_mode: str = field(
        default="disabled",
        metadata={"choices": ["online", "offline", "disabled"],
                  "help": "W&B logging mode: stream online, buffer offline, or disable."})
    tensorboard_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for TensorBoard event files; unset disables TensorBoard."})
    tensorboard_queue_size: int = field(
        default=1000,
        metadata={"help": "Max pending events buffered before the async TensorBoard writer flushes."})

    # ── Profiler ──
    use_pytorch_profiler: bool = field(
        default=False,
        metadata={"help": "Enable torch.profiler to capture CPU/GPU op traces."})
    use_nsys_profiler: bool = field(
        default=False,
        metadata={"help": "Enable NVIDIA Nsight Systems (nsys) profiling range markers."})
    profile_step_start: int = field(
        default=10,
        metadata={"help": "Iteration at which profiling capture starts."})
    profile_step_end: int = field(
        default=12,
        metadata={"help": "Iteration at which profiling capture stops."})
    profile_ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "Ranks on which the profiler is active."})
    profile_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to write profiler traces to."})

    # ── CUDA Graph ──
    cuda_graph_impl: str = field(
        default="none",
        metadata={"choices": ["none", "local"],
                  "help": "CUDA graph capture backend: 'none' disables, 'local' captures the "
                          "training step to cut per-step launch overhead."})
    cuda_graph_scope: str = field(
        default="full_iteration",
        metadata={"choices": ["full_iteration", "per_microbatch"],
                  "help": "What to capture into the graph: the whole iteration or each micro-batch."})
    cuda_graph_warmup_steps: int = field(
        default=3,
        metadata={"help": "Number of eager (uncaptured) warmup iterations before graph capture."})
    cuda_graph_pad_length: Optional[int] = field(
        default=None,
        metadata={"help": "Fixed token sequence length to pad to; required so captured shapes "
                          "stay static across steps."})
    cuda_graph_ddp_sync_in_graph: bool = field(
        default=False,
        metadata={"help": "Capture DDP gradient all-reduce inside the graph instead of running it eagerly."})
    cuda_graph_grad_sync_bucket_mb: float = field(
        default=200.0,
        metadata={"help": "Bucket size (MiB) for the manual gradient all-reduce used with CUDA graphs."})
    cuda_graph_grad_sync_impl: str = field(
        default="coalesced",
        metadata={"choices": ["flat", "coalesced"],
                  "help": "Manual gradient all-reduce implementation: single flat buffer or coalesced buckets."})
    cuda_graph_grad_sync_dtype: str = field(
        default="fp32",
        metadata={"choices": ["fp32", "bf16"],
                  "help": "Communication dtype for the manual gradient all-reduce (bf16 halves comm volume)."})
    check_for_nan_in_loss_and_grad: bool = field(
        default=True,
        metadata={"help": "Run host-side NaN/Inf checks on loss and gradients each step (small overhead)."})

    # ── Distributed ──
    distributed_strategy: str = field(
        default="fsdp",
        metadata={"choices": ["ddp", "fsdp"],
                  "help": "Parallelism strategy: DDP (replicate) or FSDP2 (fully sharded)."})
    hsdp_shard_size: Optional[int] = field(
        default=None,
        metadata={"cli_type": parse_positive_int,
                  "help": "HSDP sharding group size (second mesh dim); enables hybrid shard/replicate."})
    fsdp_reshard_default: Any = field(
        default=None,
        metadata={"cli_type": parse_reshard_after_forward,
                  "help": "Default FSDP2 reshard_after_forward policy: true|false|none|int>1. "
                          "Controls whether params are re-sharded after forward to save memory."})
    fsdp_reshard_root: Any = field(
        default=False,
        metadata={"cli_type": parse_reshard_after_forward,
                  "help": "reshard_after_forward policy for the root FSDP group specifically."})
    fsdp_reshard_module_overrides: Any = field(
        default=None,
        metadata={"cli_type": parse_reshard_after_forward_map,
                  "help": "Per-module reshard overrides as 'ClassName=value,...'."})
    fsdp_wrap_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated module class names to wrap as individual FSDP units."})
    fsdp_no_wrap_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated module class names to exclude from FSDP wrapping."})
    fsdp_min_num_params: int = field(
        default=1_000_000,
        metadata={"help": "Minimum parameter count for auto-wrapping repeated transformer layers."})
    fsdp_leftover_min_num_params: int = field(
        default=1_000_000,
        metadata={"help": "Minimum parameter count for auto-wrapping remaining (leftover) modules."})
    ddp_broadcast_buffers: bool = field(
        default=True,
        metadata={"help": "Broadcast module buffers (e.g. BN stats) from rank 0 each forward."})
    ddp_init_sync: bool = field(
        default=True,
        metadata={"help": "Synchronize parameters and buffers across ranks at initialization."})
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={"help": "Gradient all-reduce bucket size (MiB) for DDP."})
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={"help": "Detect parameters unused in the forward graph (needed for conditional "
                          "branches; adds overhead)."})
    ddp_gradient_as_bucket_view: bool = field(
        default=False,
        metadata={"help": "Expose gradients as views into DDP communication buckets to save memory."})
    ddp_static_graph: bool = field(
        default=False,
        metadata={"help": "Assume a static graph across iterations to enable DDP optimizations."})
    ddp_skip_all_reduce_unused_params: bool = field(
        default=False,
        metadata={"help": "Skip the gradient all-reduce for parameters detected as unused."})
    ddp_bucket_cap_mb_list: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated per-bucket sizes (MiB) for fine-grained DDP bucketing."})
    ddp_batched_grad_copy: bool = field(
        default=False,
        metadata={"help": "Batch gradient copies into buckets to reduce kernel launches."})
    dtype: str = field(
        default="bfloat16",
        metadata={"choices": ["bfloat16", "float16", "float32"],
                  "help": "Compute/parameter dtype for training."})
    zero_optimizer: bool = field(
        default=False,
        metadata={"help": "Shard optimizer state across ranks (ZeRO-1); DDP strategy only."})
    zero_parameters_as_bucket_view: bool = field(
        default=False,
        metadata={"help": "ZeRO parameters_as_bucket_view — alias gradients to bucket memory to save RAM."})


# ═══════════════════════════════════════════════════════════════════
# CLI generation — reflect TrainingArgs into an argparse parser
# ═══════════════════════════════════════════════════════════════════


def _base_type(field_type):
    """Resolve Optional[X] / Union[X, None] to X; leave others unchanged."""
    origin = get_origin(field_type)
    if origin is Union:
        non_none = [t for t in get_args(field_type) if t is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return field_type


def add_args_from_dataclass(parser: argparse.ArgumentParser, cls, prefix: str = ""):
    """Register one argparse argument per dataclass field.

    - ``bool`` fields use ``BooleanOptionalAction`` (``--flag`` / ``--no-flag``).
    - ``list`` fields use ``nargs='+'`` with the element type.
    - ``metadata['cli_type']`` overrides the parser for special types.
    - ``metadata['choices']`` / ``metadata['help']`` are forwarded.
    - Every arg uses ``default=SUPPRESS`` so only user-provided values appear.
    """
    for f in dataclasses.fields(cls):
        name = f"--{(prefix + f.name).replace('_', '-')}"
        meta = f.metadata
        kwargs = {
            "default": argparse.SUPPRESS,
            "help": meta.get("help", ""),
            "dest": prefix + f.name,
        }
        if "choices" in meta:
            kwargs["choices"] = meta["choices"]

        ftype = _base_type(f.type)

        if "cli_type" in meta:
            kwargs["type"] = meta["cli_type"]
        elif ftype is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif get_origin(ftype) in (list, tuple) or ftype in (list, tuple):
            elem_types = [t for t in get_args(ftype) if t is not Ellipsis]
            kwargs["type"] = elem_types[0] if elem_types else str
            kwargs["nargs"] = "+"
        else:
            kwargs["type"] = ftype

        parser.add_argument(name, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training arg parser: TrainingArgs flags + YAML dotlist overrides."""
    parser = argparse.ArgumentParser(
        description="LoongForge Embodied Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args_from_dataclass(parser, TrainingArgs)
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="YAML overrides in dotlist format: model.action_horizon=64 data.image_size=448",
    )
    return parser


__all__ = [
    "TrainingArgs",
    "add_args_from_dataclass",
    "build_arg_parser",
    "parse_reshard_after_forward",
    "parse_reshard_after_forward_map",
    "parse_positive_int",
]
