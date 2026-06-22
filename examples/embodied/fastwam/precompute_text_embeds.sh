#!/usr/bin/env bash
# Precompute FastWAM text embeddings for LoongForge training.

set -euo pipefail

export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"

DATASET_PATH="${DATASET_PATH:-/path/to/libero}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-$LOONGFORGE_PATH/data/fastwam_text_embeds}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
CONTEXT_LEN="${CONTEXT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  python "$LOONGFORGE_PATH/loongforge/embodied/data/transforms/fastwam/precompute_text_embeds.py" \
    --dataset-root "$DATASET_PATH" \
    --output-dir "$TEXT_EMBEDDING_CACHE_DIR" \
    --model-id "$MODEL_ID" \
    --tokenizer-model-id "$TOKENIZER_MODEL_ID" \
    --context-len "$CONTEXT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "$@"

echo "FastWAM text embedding cache: $TEXT_EMBEDDING_CACHE_DIR"
