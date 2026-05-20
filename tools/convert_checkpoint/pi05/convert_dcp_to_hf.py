# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Convert DCP (Distributed Checkpoint) format back to HuggingFace safetensors format.

This script reverses the conversion performed by convert_hf_to_Mfsdp.py:
loads a DCP checkpoint, strips the "module.model." prefix, and saves the
weights as safetensors files compatible with HuggingFace.

The conversion pipeline is: DCP -> intermediate .pt -> strip prefix -> safetensors.

Usage:
    python convert_dcp_to_hf.py \
        --input /path/to/dcp_checkpoint/ \
        --output /path/to/hf_output/

    # Keep intermediate .pt file
    python convert_dcp_to_hf.py \
        --input /path/to/dcp_checkpoint/ \
        --output /path/to/hf_output/ \
        --keep-pt

    # With dtype conversion
    python convert_dcp_to_hf.py \
        --input /path/to/dcp_checkpoint/ \
        --output /path/to/hf_output/ \
        --dtype bf16
"""

import argparse
import os
import sys
import tempfile
from collections import OrderedDict

import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors library required. Install with: pip install safetensors")
    sys.exit(1)

try:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
except ImportError:
    print("ERROR: torch.distributed.checkpoint not available. Requires PyTorch >= 2.3")
    sys.exit(1)

from convert_torch_to_hf import (
    DEFAULT_MAX_SHARD_SIZE,
    convert_dtype,
    parse_shard_size,
    print_summary,
    save_safetensors,
    shard_state_dict,
    strip_prefix,
)


def load_dcp_checkpoint(dcp_dir: str, keep_pt: bool = False) -> OrderedDict:
    """Load a DCP checkpoint by converting to a temporary .pt file first."""
    if keep_pt:
        pt_path = os.path.join(dcp_dir, "_dcp_converted.pt")
    else:
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        pt_path = temp_file.name
        temp_file.close()

    print(f"Converting DCP to .pt: {dcp_dir} -> {pt_path}")
    dcp_to_torch_save(dcp_dir, pt_path)

    print(f"Loading converted .pt file: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"Loaded DCP checkpoint (iteration={checkpoint.get('iteration', '?')}, "
              f"checkpoint_version={checkpoint.get('checkpoint_version', '?')})")
    elif isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
        print("Loaded plain state_dict (no envelope)")
    else:
        print(f"ERROR: Unexpected checkpoint format: {type(checkpoint)}")
        sys.exit(1)

    if not keep_pt:
        os.remove(pt_path)
        print(f"Removed temporary file: {pt_path}")

    return state_dict


def build_arg_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert Pi05 DCP (Distributed Checkpoint) to HuggingFace safetensors"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to DCP checkpoint directory"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for safetensors files"
    )
    parser.add_argument(
        "--prefix-remove", type=str, default="module.model.",
        help="Prefix to remove from parameter names (default: 'module.model.')"
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
        "--keep-pt", action="store_true",
        help="Keep intermediate .pt file after conversion"
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

    # 1. Load DCP checkpoint
    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    state_dict = load_dcp_checkpoint(args.input, keep_pt=args.keep_pt)
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
