#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# run_fast_ddp.sh - fast VLA Training Launch Script (DDP, Single Node)
#
# Usage:
#   bash run_fast_ddp.sh                                        # paligemma default
#   bash run_fast_ddp.sh --lr 1e-4                              # override training param
#   bash run_fast_ddp.sh backbone.image_size=448                # override YAML field
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH="/workspace/20260604_master/AIAK-Training-Omni"

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/model/fast_model"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/model/Qwen2.5-VL-3B-Instruct-Action"}
BASE_VLM_PATH=${BASE_VLM_PATH:-"/workspace/model/Qwen2.5-VL-3B-Instruct-Action"}
DATA_PATH=${DATA_PATH:-"/workspace/libero_spatial_no_noops_1.0.0_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/qwenfast_$(date +%Y%m%d_%H%M%S)"}

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"qwen_fast"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataloader-module lerobot_datasets
    --dataset-path $DATA_PATH
    --lerobotdataset-version v2.0
    --tokenizer-path $TOKENIZER_PATH
    --robot-type libero_franka
    --normalization-mode q99
    --num-workers 4
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --training-phase finetune
    --trainer-type BCTrainer
    --train-iters 20
    --per-device-batch-size 4
    --gradient-accumulation-steps 2
    --seed 42
    --output-dir $OUTPUT_DIR
    # Learning rate
    --lr 2.5e-5
    --lr-backbone 1.0e-5
    --lr-action-model 1.0e-4
    --lr-decay-style cosine_with_min_lr
    --lr-warmup-iters 10
    --min-lr 1.0e-6
    # Optimizer
    --optimizer AdamW
    --clip-grad 1.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # EMA (disabled)
    # --ema
    --ema-decay 0.9999
    # Checkpoint
    --save-interval 10
    # --pretrained-checkpoint $CHECKPOINT_PATH  
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
echo "  LoongForgeVLA QwenFAST Training (DDP)"
echo "  Model:      $MODEL_NAME"
echo "  Base VLM:   $BASE_VLM_PATH"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "backbone.base_vlm=$BASE_VLM_PATH" \
    "$@"   # pass-through: --lr 1e-4  OR  backbone.image_size=448
