#!/bin/bash
# Pi0.5 Flow Matching VLA Training — DDP, Single Node

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$PROJECT_ROOT"}
EMBODIED_ROOT="$PROJECT_ROOT/loongforge/embodied"

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-"/ssd1/yexiaochuan/pi05_base/"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/ssd1/yexiaochuan/paligemma-3b-pt-224"}

DATASET_PATH=${DATASET_PATH:-"/ssd1/yexiaochuan/libero"}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-"/ssd1/yexiaochuan"}

OUTPUT_DIR=${OUTPUT_DIR:-"outputs/pi05_ddp_$(date +%Y%m%d_%H%M%S)"}

GPUS_PER_NODE=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29501}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes 1
  --master_port $MASTER_PORT
)

TRAINING_ARGS=(
  --model-name pi05_paligemma
  --training-phase finetune
  --distributed-strategy ddp
  --dtype bfloat16
  --max-train-steps 20
  --save-steps 10
  --gradient-clipping 1.0
  --gradient-accumulation-steps 2
  --logging-frequency 50
  --seed 3047
  --output-dir $OUTPUT_DIR
  --pretrained-checkpoint $PRETRAINED_CHECKPOINT
)

LR_ARGS=(
  --lr 2.5e-05
  --lr-backbone 1.0e-05
  --lr-action-model 1.0e-04
  --lr-scheduler-type cosine_with_min_lr
  --warmup-steps 2000
  --min-lr 1.0e-06
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.95
)

EMA_ARGS=(
  --ema
  --ema-decay 0.9999
)

DATA_ARGS=(
  --dataloader-module lerobot_datasets
  --data-root-dir $DATA_ROOT_DIR
  --dataset-path $DATASET_PATH
  --robot-type libero_franka
  --per-device-batch-size 4
  --num-workers 4
  --action-horizon 10
  --image-size 224
  --normalization-mode q99
)

LOGGING_ARGS=(
  --wandb-project loongforge-vla
  --wandb-mode disabled
)

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $EMBODIED_ROOT/train.py \
  ${TRAINING_ARGS[@]} \
  ${LR_ARGS[@]} \
  ${EMA_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  "$@"
