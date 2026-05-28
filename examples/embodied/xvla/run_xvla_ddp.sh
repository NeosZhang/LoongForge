#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# run_pi05_ddp.sh - π₀.₅ VLA Training Launch Script (DDP, Single Node)
#
# Usage:
#   bash run_pi05_ddp.sh                                        # paligemma default
#   bash run_pi05_ddp.sh --lr 1e-4                              # override training param
#   bash run_pi05_ddp.sh backbone.image_size=448                # override YAML field
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH="/workspace/AIAK-Training-Omni"

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cfs/pxy/ckpt/facebook/bart-large"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cfs/pxy/ckpt/xvla-base"}
export DATA_PATH=${DATA_PATH:-"/mnt/cfs/pxy/data/libero"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/cfs/pxy/outputs/xvla_ddp_$(date +%Y%m%d_%H%M%S)"}
MASTER_PORT=29235
GRADIENT_ACCUMULATION_STEPS=1
#export XVLA_PROFILE_TIME=1
#export EMBODIED_TRAIN_PROFILE_SYNC=1

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=1
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}
echo "GPUS_PER_NODE="$GPUS_PER_NODE
echo "NNODES="$NNODES

mkdir -p ${OUTPUT_DIR}/nsys_report


NSYS_ARGS="nsys profile \
    --output=${OUTPUT_DIR}/nsys_report \
    -s none --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    -- "

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"xvla"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataloader-module lerobot_datasets
    --dataset-path $DATA_PATH
    --tokenizer-path $TOKENIZER_PATH
    --robot-type libero_franka
    --normalization-mode q99
    --num-workers 4
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --training-phase finetune
    --trainer-type BCTrainer
    --train-iters 1000
    --per-device-batch-size 4
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed 42
    --output-dir $OUTPUT_DIR
    # Learning rate
    --lr 0.0001
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 10
    --loss-spike-threshold 1000
    # Optimizer
    --optimizer AdamW
    --clip-grad 10
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # EMA
    #--ema
    #--ema-decay 0.9999
    # Checkpoint
    --save-interval 10
    --pretrained-checkpoint $CHECKPOINT_PATH
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 50
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForgeVLA X-VLA Training (DDP)"
echo "  Model:      $MODEL_NAME"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    $NSYS \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"   # pass-through: --lr 1e-4  OR  backbone.image_size=448