#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Convert Pi05 PyTorch (Megatron DDP) checkpoint to HuggingFace format.
#
# This script converts a Megatron-format .pt checkpoint back to HuggingFace
# safetensors format, reversing the operation of convert_pi05_hf_to_torch.sh.
#
# Usage:
#   # Using default paths (edit LOAD/SAVE below)
#   bash convert_pi05_torch_to_hf.sh
#
#   # Using environment variables
#   LOAD=/path/to/torch_ckpt SAVE=/path/to/output bash convert_pi05_torch_to_hf.sh

set -euo pipefail

LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}

# Input/Output paths - modify as needed
LOAD=${LOAD:-"/workspace/pi05_torch/"}
SAVE=${SAVE:-"/workspace/pi05_huggingface/"}

# Conversion options
DTYPE=${DTYPE:-""}
PREFIX_REMOVE=${PREFIX_REMOVE:-"model."}
MAX_SHARD_SIZE=${MAX_SHARD_SIZE:-"5GB"}

echo "=========================================="
echo "Pi05 PyTorch to HuggingFace Convert"
echo "=========================================="
echo "Input:  ${LOAD}"
echo "Output: ${SAVE}"
echo "Prefix: ${PREFIX_REMOVE}"
echo "=========================================="

# Find the .pt file
PT_FILE="${LOAD}/release/mp_rank_00/model_optim_rng.pt"
if [[ ! -f "${PT_FILE}" ]]; then
    echo "ERROR: Checkpoint not found at ${PT_FILE}"
    exit 1
fi

# Build optional args
EXTRA_ARGS=()
if [[ -n "${DTYPE}" ]]; then
    EXTRA_ARGS+=("--dtype" "${DTYPE}")
fi

# Run conversion
"${PYTHON_BIN}" "${LOONGFORGE_PATH}/tools/convert_checkpoint/pi05/convert_torch_to_hf.py" \
    --input "${PT_FILE}" \
    --output "${SAVE}" \
    --prefix-remove "${PREFIX_REMOVE}" \
    --max-shard-size "${MAX_SHARD_SIZE}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "HuggingFace checkpoint saved to: ${SAVE}"
echo "=========================================="
