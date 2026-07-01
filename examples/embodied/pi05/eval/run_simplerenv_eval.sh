#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/LoongForge-VLA}
EVAL_ROOT=${EVAL_ROOT:-${REPO_ROOT}/loongforge/embodied/eval}
EXAMPLE_EVAL_ROOT=${EXAMPLE_EVAL_ROOT:-${REPO_ROOT}/examples/embodied/pi05/eval}
CONFIG=${CONFIG:-${EXAMPLE_EVAL_ROOT}/configs/simplerenv/eggplant_300step.yaml}
if [[ "${CONFIG}" != /* ]]; then
  CONFIG=${REPO_ROOT}/${CONFIG}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/path/to/nvidia_icd.json}

/workspace/miniconda3/envs/simplerenv/bin/python -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}"
