#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EVAL_ROOT=${EVAL_ROOT:-${REPO_ROOT}/loongforge/embodied/eval}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/pi05/eval}
CONFIG=${EXAMPLE_EVAL_ROOT}/configs/robotwin/random_init_5step_internal.yaml

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/ssd1/opt/nvidia_lib:/usr/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/ssd1/opt/nvidia_lib/10_nvidia.json}

/workspace/miniconda3/envs/robotwin/bin/python -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
