#!/bin/bash
# Extract token-level router embeddings with shuffled sampling and per-document truncation.
#
# Like run_extract_embeddings_shuffled_token.sh, but truncates each document to
# --max-tokens-per-doc tokens. This increases document diversity at a fixed token
# budget (e.g. 1M tokens / 100 tokens per doc = ~10K documents instead of ~150).
#
# Prerequisites: mix_composition.json must exist (run run_analyze_data_mix.sh first)
#
# Usage:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token_truncated.sh [MODEL_PATH] [TARGET_TOKENS] [MAX_TOKENS_PER_DOC]
#
# Examples:
#   # Default: 1M tokens, 100 tokens/doc, both models
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token_truncated.sh
#
#   # Custom model
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token_truncated.sh models/moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf
#
# Output goes to: claude_outputs/analysis/router_clustering_pretraining_shuffled_token_truncated/<model_name>/
set -euo pipefail

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf}"
TARGET_POST_TRUNC_TOKENS="${2:-1000000}"
MAX_TOKENS_PER_DOC="${3:-100}"

# --target-tokens controls pre-truncation doc loading. Average doc is ~680 tokens,
# so we need ~7x more pre-truncation tokens to get the desired post-truncation count.
# Using 8x for safety margin.
TARGET_TOKENS=$(( TARGET_POST_TRUNC_TOKENS * 8 ))

# Derive model name (parent directory of the checkpoint)
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/analysis/router_clustering_pretraining_shuffled_token_truncated"
COMPOSITION_FILE="claude_outputs/analysis/router_clustering_pretraining/mix_composition.json"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"
BATCH_SIZE=32

echo "Model: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Target post-truncation tokens: $TARGET_POST_TRUNC_TOKENS"
echo "Target pre-truncation tokens: $TARGET_TOKENS (8x oversampling)"
echo "Max tokens per doc: $MAX_TOKENS_PER_DOC"
echo "Sampling: SHUFFLED (random offsets across all files)"

if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "ERROR: ${COMPOSITION_FILE} not found. Run run_analyze_data_mix.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Token-level extraction (truncated): ${MODEL_NAME} ==="
PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --granularity token \
    --target-tokens "$TARGET_TOKENS" \
    --max-tokens-per-doc "$MAX_TOKENS_PER_DOC" \
    --batch-size "$BATCH_SIZE" \
    --shuffle \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy "${OUTPUT_DIR}"/documents.npy "${OUTPUT_DIR}"/doc_boundaries.npy
