# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Convert PyTorch (Megatron DDP) checkpoint back to HuggingFace safetensors format.

This script reverses the conversion performed by convert_hf_to_torch.py:
loads a Megatron-format .pt checkpoint, strips the "model." prefix, and
saves the weights as safetensors files compatible with HuggingFace.

Usage:
    python convert_torch_to_hf.py \
        --input /path/to/release/mp_rank_00/model_optim_rng.pt \
        --output /path/to/hf_output/

    # With dtype conversion
    python convert_torch_to_hf.py \
        --input /path/to/model_optim_rng.pt \
        --output /path/to/hf_output/ \
        --dtype bf16

    # Dry run to check keys without saving
    python convert_torch_to_hf.py \
        --input /path/to/model_optim_rng.pt \
        --dry-run
"""

import argparse
import os
import sys
from collections import OrderedDict

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors library required. Install with: pip install safetensors")
    sys.exit(1)


# Default shard size (same as HuggingFace transformers)
DEFAULT_MAX_SHARD_SIZE = "5GB"


def parse_shard_size(size_str: str) -> int:
    """Parse human-readable shard size string to bytes."""
    size_str = size_str.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * mult)
    return int(size_str)


def load_torch_checkpoint(pt_path: str) -> OrderedDict:
    """Load a Megatron-format .pt checkpoint and extract model weights."""
    print(f"Loading checkpoint: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"Loaded Megatron-format checkpoint (iteration={checkpoint.get('iteration', '?')}, "
              f"checkpoint_version={checkpoint.get('checkpoint_version', '?')})")
    elif isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
        print("Loaded plain state_dict (no Megatron envelope)")
    else:
        print(f"ERROR: Unexpected checkpoint format: {type(checkpoint)}")
        sys.exit(1)

    if not isinstance(state_dict, dict):
        print(f"ERROR: state_dict is not a dict: {type(state_dict)}")
        sys.exit(1)

    return state_dict


def strip_prefix(state_dict: OrderedDict, prefix: str) -> OrderedDict:
    """Remove a prefix from all keys in the state dict."""
    new_sd = OrderedDict()
    stripped = 0
    not_stripped = 0
    for key, val in state_dict.items():
        if key.startswith(prefix):
            new_sd[key[len(prefix):]] = val
            stripped += 1
        else:
            new_sd[key] = val
            not_stripped += 1
    print(f"Stripped prefix '{prefix}': {stripped} keys affected, {not_stripped} keys unchanged")
    return new_sd


def convert_dtype(state_dict: OrderedDict, dtype_str: str) -> OrderedDict:
    """Convert all floating-point tensors to the target dtype."""
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    target_dtype = dtype_map[dtype_str]
    converted = 0
    for key in state_dict:
        if state_dict[key].is_floating_point() and state_dict[key].dtype != target_dtype:
            state_dict[key] = state_dict[key].to(target_dtype)
            converted += 1
    print(f"Converted {converted} tensors to {dtype_str}")
    return state_dict


def print_summary(state_dict: OrderedDict, show_all: bool = False):
    """Print state_dict summary."""
    total_params = len(state_dict)
    total_elements = 0
    dtype_counts = {}

    for name, tensor in state_dict.items():
        total_elements += tensor.numel()
        dt = str(tensor.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    print(f"\n{'='*60}")
    print("State Dict Summary:")
    print(f"{'='*60}")
    print(f"  Number of tensors:    {total_params}")
    print(f"  Total elements:       {total_elements:,}")
    print(f"  BF16 size estimate:   {total_elements * 2 / 1024**3:.2f} GB")
    print(f"  FP32 size estimate:   {total_elements * 4 / 1024**3:.2f} GB")
    print(f"  Data type distribution: {dtype_counts}")

    if show_all:
        print(f"\n{'='*60}")
        print("All Parameters:")
        print(f"{'='*60}")
        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            print(f"  {name:80s}  {str(t.shape):>30s}  {t.dtype}")


def shard_state_dict(state_dict: OrderedDict, max_shard_size: int) -> list[OrderedDict]:
    """Split state_dict into shards based on max shard size."""
    shards = []
    current_shard = OrderedDict()
    current_size = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_shard and current_size + tensor_size > max_shard_size:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    return shards


def save_safetensors(state_dict: OrderedDict, output_dir: str, max_shard_size: str):
    """Save state_dict as safetensors, sharding if necessary."""
    os.makedirs(output_dir, exist_ok=True)
    max_bytes = parse_shard_size(max_shard_size)
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())

    if total_size <= max_bytes:
        # Single file
        out_path = os.path.join(output_dir, "model.safetensors")
        print(f"Saving to single file: {out_path}")
        save_file(state_dict, out_path)
        size_mb = os.path.getsize(out_path) / 1024**2
        print(f"Save complete! File size: {size_mb:.1f} MB")
    else:
        # Sharded
        shards = shard_state_dict(state_dict, max_bytes)
        num_shards = len(shards)
        print(f"Saving as {num_shards} shards (max shard size: {max_shard_size}):")

        # Build index
        index = {"metadata": {"total_size": total_size}, "weight_map": {}}

        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
            out_path = os.path.join(output_dir, shard_name)
            print(f"  [{i+1}/{num_shards}] {shard_name} ({len(shard)} tensors)")
            save_file(shard, out_path)

            for key in shard:
                index["weight_map"][key] = shard_name

        # Write index JSON
        import json
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Index written to: {index_path}")


def build_arg_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert Pi05 PyTorch (Megatron DDP) checkpoint to HuggingFace safetensors"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to .pt checkpoint file (e.g., release/mp_rank_00/model_optim_rng.pt)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for safetensors files"
    )
    parser.add_argument(
        "--prefix-remove", type=str, default="model.",
        help="Prefix to remove from parameter names (default: 'model.')"
    )
    parser.add_argument(
        "--dtype", type=str, default=None, choices=["fp32", "fp16", "bf16"],
        help="Target data type (default: preserve original)"
    )
    parser.add_argument(
        "--max-shard-size", type=str, default=DEFAULT_MAX_SHARD_SIZE,
        help=f"Maximum shard size (default: {DEFAULT_MAX_SHARD_SIZE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print key list and summary only, do not save"
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print all parameter names and shapes"
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    echo_args = {k: v for k, v in vars(args).items() if k not in ("dry_run", "show_all")}
    print(f"Arguments: {echo_args}")

    # 1. Load checkpoint
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    state_dict = load_torch_checkpoint(args.input)
    print(f"Loaded {len(state_dict)} parameters")

    # 2. Strip prefix
    if args.prefix_remove:
        state_dict = strip_prefix(state_dict, args.prefix_remove)

    # 3. Optional: dtype conversion
    if args.dtype:
        state_dict = convert_dtype(state_dict, args.dtype)

    # 4. Print summary
    print_summary(state_dict, show_all=args.show_all or args.dry_run)

    # 5. Save
    if args.dry_run:
        print("\n[dry-run] Not saving file")
    else:
        save_safetensors(state_dict, args.output, args.max_shard_size)
        print(f"\nConversion complete! Output: {args.output}")


if __name__ == "__main__":
    main()
