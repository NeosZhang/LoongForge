#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Convert Pi05 DCP (Distributed Checkpoint) to HuggingFace format.
#
# This script converts a DCP checkpoint back to HuggingFace safetensors format,
# reversing the operation of convert_pi05_hf_to_dcp.sh.
#
# Usage:
#   # Using default paths (edit LOAD/SAVE below)
#   bash convert_pi05_dcp_to_hf.sh
#
#   # Using environment variables
#   LOAD=/path/to/dcp_ckpt SAVE=/path/to/output bash convert_pi05_dcp_to_hf.sh

set -euo pipefail

LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}

# Input/Output paths - modify as needed
LOAD=${LOAD:-"/workspace/pi05_omni/release/"}
SAVE=${SAVE:-"/workspace/pi05_huggingface/"}

# Conversion options
DTYPE=${DTYPE:-""}
PREFIX_REMOVE=${PREFIX_REMOVE:-"module.model."}
MAX_SHARD_SIZE=${MAX_SHARD_SIZE:-"5GB"}
KEEP_PT=${KEEP_PT:-"false"}

echo "=========================================="
echo "Pi05 DCP to HuggingFace Convert"
echo "=========================================="
echo "Input:  ${LOAD}"
echo "Output: ${SAVE}"
echo "Prefix: ${PREFIX_REMOVE}"
echo "=========================================="

if [[ ! -d "${LOAD}" ]]; then
    echo "ERROR: DCP checkpoint directory not found: ${LOAD}"
    exit 1
fi

# Build optional args
EXTRA_ARGS=()
if [[ -n "${DTYPE}" ]]; then
    EXTRA_ARGS+=("--dtype" "${DTYPE}")
fi
if [[ "${KEEP_PT}" == "true" ]]; then
    EXTRA_ARGS+=("--keep-pt")
fi

# Run conversion
"${PYTHON_BIN}" "${LOONGFORGE_PATH}/tools/convert_checkpoint/pi05/convert_dcp_to_hf.py" \
    --input "${LOAD}" \
    --output "${SAVE}" \
    --prefix-remove "${PREFIX_REMOVE}" \
    --max-shard-size "${MAX_SHARD_SIZE}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "HuggingFace checkpoint saved to: ${SAVE}"
echo "=========================================="
