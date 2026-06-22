#!/usr/bin/env bash
# Preprocess ActionDiT backbone weights from Wan2.2 video DiT for LoongForge FastWAM.

set -euo pipefail

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
OUTPUT="${OUTPUT:-$LOONGFORGE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
# Local root directory for model artifacts; files are looked up at <LOCAL_MODEL_PATH>/<model-id>/.
# Example: LOCAL_MODEL_PATH=/data/models 
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-}"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  python "$LOONGFORGE_PATH/loongforge/embodied/data/transforms/fastwam/preprocess_action_dit_backbone.py" \
    --output "$OUTPUT" \
    --model-id "$MODEL_ID" \
    --tokenizer-model-id "$TOKENIZER_MODEL_ID" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --local-model-path "$LOCAL_MODEL_PATH" \
    "$@"

echo "ActionDiT backbone payload saved to: $OUTPUT"
