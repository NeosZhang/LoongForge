#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_pi05.sh - π₀.₅ Flow Matching VLA Training Launch Script
#
# Supports multiple VLM backbones:
#   - PaliGemma 3B + Gemma Action Expert (original Pi0.5)
#   - Qwen2.5-VL 3B + Gemma Action Expert (prefix-cache mode)
#   - Llama 3.2 Vision 11B + Gemma Action Expert (prefix-cache mode)
#
# Usage:
#   # PaliGemma Pi0.5 (LIBERO all tasks, 8 GPUs)
#   bash scripts/run_pi05.sh paligemma
#
#   # Qwen Pi0.5 (LIBERO spatial, 4 GPUs)
#   NUM_GPUS=4 bash scripts/run_pi05.sh qwen
#
#   # Custom config file
#   bash scripts/run_pi05.sh custom configs/my_pi05.yaml
#
#   # Additional overrides
#   bash scripts/run_pi05.sh paligemma \
#       trainer.max_train_steps=200000 \
#       datasets.vla_data.dataset_mix=libero_all
#
#   # Resume from checkpoint
#   bash scripts/run_pi05.sh paligemma \
#       trainer.is_resume=true \
#       trainer.pretrained_checkpoint=outputs/paligemma_pi05_libero/checkpoint-50000
# ═══════════════════════════════════════════════════════════════

export CUDA_VISIBLE_DEVICES=0
export USE_DDP=1
export TOKENIZER_PATH=/ssd1/yexiaochuan/paligemma-3b-pt-224
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EMBODIED_ROOT="$PROJECT_ROOT/loongforge/embodied"
cd "$PROJECT_ROOT"

# ── Argument Parsing ──
VLM_TYPE="${1:-paligemma}"
shift || true
EXTRA_ARGS="$@"

# ── Config Mapping ──
case "$VLM_TYPE" in
    paligemma|pg)
        CONFIG_FILE="$PROJECT_ROOT/configs/models/embodied/paligemma_pi05.yaml"
        RUN_NAME="paligemma_pi05"
        ;;
    qwen|qwen2.5)
        CONFIG_FILE="$EMBODIED_ROOT/configs/qwen_pi05.yaml"
        RUN_NAME="qwen_pi05"
        ;;
    custom)
        CONFIG_FILE="${1:?Usage: $0 custom <config.yaml> [overrides...]}"
        shift
        EXTRA_ARGS="$@"
        RUN_NAME="custom_pi05"
        ;;
    *)
        echo "Error: Unknown VLM type '$VLM_TYPE'"
        echo "Supported: paligemma, qwen, custom"
        exit 1
        ;;
esac

# ── Environment Variables ──
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    _GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
fi
NUM_GPUS=1
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/deepspeed/accelerate_zero2.yaml}"
ACCELERATE_CONFIG=""
MASTER_PORT="${MASTER_PORT:-29500}"

# ── Training Parameters (override via env or edit below) ──
# Trainer config
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-150000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
GRADIENT_CLIPPING="${GRADIENT_CLIPPING:-1.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"
FREEZE_MODULES="${FREEZE_MODULES:-vlm_interface.model.language_model}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-05}"
LR_VLM="${LR_VLM:-1.0e-05}"
LR_ACTION_MODEL="${LR_ACTION_MODEL:-1.0e-04}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine_with_min_lr}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
MIN_LR="${MIN_LR:-1.0e-06}"
EMA_DECAY="${EMA_DECAY:-0.9999}"

# Running config
SEED="${SEED:-3047}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_ID="${RUN_ID:-$RUN_NAME}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-loongforge-vla}"
WANDB_MODE="${WANDB_MODE:-online}"

# Dataset config
DATALOADER_MODULE="${DATALOADER_MODULE:-lerobot_datasets}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data/lerobot}"
DATASET_PATH="${DATASET_PATH:-/ssd1/yexiaochuan/libero}"
#DATASET_MIX="${DATASET_MIX:-libero_all}"
ROBOT_TYPE="${ROBOT_TYPE:-libero_franka}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NORMALIZATION_MODE="${NORMALIZATION_MODE:-q99}"

# Build CLI args
TRAIN_ARGS=(
    --config "$CONFIG_FILE"
    # Trainer
    --max_train_steps "$MAX_TRAIN_STEPS"
    --save_steps "$SAVE_STEPS"
    --gradient_clipping "$GRADIENT_CLIPPING"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --logging_frequency "$LOGGING_FREQUENCY"
    --freeze_modules "$FREEZE_MODULES"
    --learning_rate "$LEARNING_RATE"
    --lr_vlm "$LR_VLM"
    --lr_action_model "$LR_ACTION_MODEL"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --warmup_steps "$WARMUP_STEPS"
    --min_lr "$MIN_LR"
    --ema_enabled
    --ema_decay "$EMA_DECAY"
    # Running
    --seed "$SEED"
    --output_dir "$OUTPUT_DIR"
    --run_id "$RUN_ID"
    --wandb_project "$WANDB_PROJECT_NAME"
    --wandb_mode "$WANDB_MODE"
    # Dataset
    --dataloader_module "$DATALOADER_MODULE"
    --data_root_dir "$DATA_ROOT_DIR"
    --dataset_path "$DATASET_PATH"
    #--dataset_mix "$DATASET_MIX"
    --robot_type "$ROBOT_TYPE"
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --action_horizon "$ACTION_HORIZON"
    --image_size "$IMAGE_SIZE"
    --normalization_mode "$NORMALIZATION_MODE"
)

echo "════════════════════════════════════════════════════════════"
echo "  LoongForgeVLA π₀.₅ Training"
echo "  VLM Type:   $VLM_TYPE"
echo "  Config:     $CONFIG_FILE"
echo "  GPUs:       $NUM_GPUS"
echo "  Accelerate: $ACCELERATE_CONFIG"
echo "  Run ID:     $RUN_ID"
#echo "  Dataset:    $DATASET_MIX"
echo "  Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "  Max Steps:  $MAX_TRAIN_STEPS"
echo "  Extra args: $EXTRA_ARGS"
echo "════════════════════════════════════════════════════════════"

# ── Environment Check ──
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: CUDA not available, falling back to CPU (very slow!)"
fi

# ── Launch Training ──
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[Launch] accelerate launch with $NUM_GPUS GPUs..."
    ACCELERATE_EXTRA_ARGS=()
    if [ -n "$ACCELERATE_CONFIG" ]; then
        ACCELERATE_EXTRA_ARGS+=(--config_file "$ACCELERATE_CONFIG")
    fi
    accelerate launch \
        "${ACCELERATE_EXTRA_ARGS[@]}" \
        --num_processes "$NUM_GPUS" \
        --main_process_port "$MASTER_PORT" \
        "$EMBODIED_ROOT/training/train.py" \
        "${TRAIN_ARGS[@]}" \
        $EXTRA_ARGS
else
    echo "[Launch] Single GPU training..."
    python "$EMBODIED_ROOT/training/train.py" \
        "${TRAIN_ARGS[@]}" \
        $EXTRA_ARGS
fi
