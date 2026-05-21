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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Argument Parsing ──
VLM_TYPE="${1:-paligemma}"
shift || true
EXTRA_ARGS="$@"

# ── Config Mapping ──
case "$VLM_TYPE" in
    paligemma|pg)
        CONFIG_FILE="configs/paligemma_pi05_layered.yaml"
        RUN_NAME="paligemma_pi05"
        ;;
    qwen|qwen2.5)
        CONFIG_FILE="configs/qwen_pi05_layered.yaml"
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
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/deepspeed/accelerate_zero2.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "════════════════════════════════════════════════════════════"
echo "  LoongForgeVLA π₀.₅ Training"
echo "  VLM Type:   $VLM_TYPE"
echo "  Config:     $CONFIG_FILE"
echo "  GPUs:       $NUM_GPUS"
echo "  Accelerate: $ACCELERATE_CONFIG"
echo "  Extra args: $EXTRA_ARGS"
echo "════════════════════════════════════════════════════════════"

# ── Environment Check ──
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: CUDA not available, falling back to CPU (very slow!)"
fi

# ── Launch Training ──
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[Launch] accelerate launch with $NUM_GPUS GPUs..."
    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        --main_process_port "$MASTER_PORT" \
        training/train_layered.py \
        --config "$CONFIG_FILE" \
        $EXTRA_ARGS
else
    echo "[Launch] Single GPU training..."
    python training/train_layered.py \
        --config "$CONFIG_FILE" \
        $EXTRA_ARGS
fi
