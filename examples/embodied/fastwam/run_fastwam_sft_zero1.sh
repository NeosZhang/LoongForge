#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# FastWAM SFT with ZeRO Stage-1 (multi-GPU DDP).
#
# Delta versus run_fastwam_sft_ddp.sh:
#   --zero-optimizer                    wrap the optimizer in
#                                       ZeroRedundancyOptimizer, sharding
#                                       optimizer states across ranks. Only
#                                       effective with --distributed-strategy ddp.
#   --no-ddp-find-unused-parameters     skip the unused-parameter scan; FastWAM's
#                                       forward graph has no conditional branches.
#   --ddp-static-graph                  graph is identical every iteration, lets
#                                       DDP reuse its bucket/reduction plan.
#   --ddp-gradient-as-bucket-view       expose grads as views into the comm
#                                       buckets instead of separate allocations.
#   --no-ddp-broadcast-buffers          no BN-style buffers to sync each forward.
#   --ddp-bucket-cap-mb                 larger buckets: fewer, bigger all-reduces.
#
# The memory saved by ZeRO-1 is what makes the larger --per-device-batch-size
# below affordable relative to the plain DDP script.
#
# Two optional ZeRO knobs are left off by default:
#   --zero-parameters-as-bucket-view    further cuts peak memory, but can clash
#                                       with torch.compile + the DDP reducer.
#   --zero-master-param-dtype fp32      rank-local fp32 master params, broadcast
#                                       after each step. Better numerics under
#                                       bf16 training at some bandwidth cost.
#
# Usage:
#   DATASET_PATH=/path/to/libero TOKENIZER_PATH=/path/to/tokenizer \
#     bash run_fastwam_sft_zero1.sh
#   GPUS_PER_NODE=4 bash run_fastwam_sft_zero1.sh   # override via env
#   ... bash run_fastwam_sft_zero1.sh --train-iters 50   # override a flag

set -euo pipefail

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

DATASET_PATH=${DATASET_PATH:-/path/to/libero}
TOKENIZER_PATH=${TOKENIZER_PATH:-/path/to/tokenizer}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/fastwam_sft_zero1_$(date +%Y%m%d_%H%M%S)"}

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH:-}
TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29519}

MODEL_NAME=${MODEL_NAME:-"fastwam"}

TRAINING_ARGS=(
  --model-name "$MODEL_NAME"
  --trainer-type FinetuneTrainer
  --dtype bfloat16
  --train-iters 20000
  --save-interval 2000
  --clip-grad 1.0
  --gradient-accumulation-steps 1
  --log-interval 1
  --seed 3047
  --output-dir "$OUTPUT_DIR"
  --tokenizer-path "$TOKENIZER_PATH"
)

# ── ZeRO Stage-1 + DDP tuning (the point of this script) ──────
ZERO1_ARGS=(
  --distributed-strategy ddp
  --zero-optimizer
  --no-ddp-find-unused-parameters
  --ddp-static-graph
  --ddp-gradient-as-bucket-view
  --no-ddp-broadcast-buffers
  --ddp-bucket-cap-mb 200
)

if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  TRAINING_ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi

LR_ARGS=(
  --lr-base 1.0e-8
  --lr-decay-style cosine_warmup_with_min_lr
  --lr-warmup-iters 0
  --min-lr 1.0e-9
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.95
)

DATA_ARGS=(
  --dataset-format lerobot_datasets
  --dataset-strategy fastwam
  --dataset-path "$DATASET_PATH"
  --robot-type libero_franka
  --per-device-batch-size 16
  --num-workers 16
  --lerobotdataset-version v2.1
  --video-backend pyav
)

LOGGING_ARGS=(
  --wandb-project loongforge-vla
  --wandb-mode disabled
)

# ── Model/data dotlist overrides ──────────────────────────────
MODEL_DATA_OVERRIDES=()
if [[ -n "$ACTION_DIT_PRETRAINED_PATH" ]]; then
  MODEL_DATA_OVERRIDES+=("model.action_dit_pretrained_path=$ACTION_DIT_PRETRAINED_PATH")
fi
if [[ -n "$TEXT_EMBEDDING_CACHE_DIR" ]]; then
  MODEL_DATA_OVERRIDES+=("data.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE_DIR")
fi

mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT (DDP + ZeRO-1)"
echo "  Model:    $MODEL_NAME"
echo "  GPUs:     $GPUS_PER_NODE"
echo "  Data:     $DATASET_PATH"
echo "  Output:   $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  torchrun --nproc_per_node "$GPUS_PER_NODE" --master_port "$MASTER_PORT" \
  "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
  "${TRAINING_ARGS[@]}" \
  "${ZERO1_ARGS[@]}" \
  "${LR_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "${MODEL_DATA_OVERRIDES[@]+"${MODEL_DATA_OVERRIDES[@]}"}" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"
