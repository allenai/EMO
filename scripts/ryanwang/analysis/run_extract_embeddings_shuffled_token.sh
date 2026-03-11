#!/bin/bash
# Extract token-level router embeddings with shuffled (random) document sampling.
#
# Extracts per-token router activations (probs, logits, topk_binary) instead of
# per-document aggregated embeddings. Uses ~100K tokens by default.
# Also saves full document token arrays for context recovery.
#
# Prerequisites: mix_composition.json must exist (run run_analyze_data_mix.sh first)
#
# Usage:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token.sh [MODEL_PATH] [TARGET_TOKENS]
#
# Examples:
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token.sh models/moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf
#   bash scripts/ryanwang/analysis/run_extract_embeddings_shuffled_token.sh models/moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf 1000000
#
# Output goes to: claude_outputs/analysis/router_clustering_pretraining_shuffled_token/<model_name>/
set -euo pipefail

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf}"
TARGET_TOKENS="${2:-100000}"

# Derive model name (parent directory of the checkpoint)
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/analysis/router_clustering_pretraining_shuffled_token"
COMPOSITION_FILE="claude_outputs/analysis/router_clustering_pretraining/mix_composition.json"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"
BATCH_SIZE=32

echo "Model: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Target tokens: $TARGET_TOKENS"
echo "Sampling: SHUFFLED (random offsets across all files)"

if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "ERROR: ${COMPOSITION_FILE} not found. Run run_analyze_data_mix.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Token-level extraction: ${MODEL_NAME} ==="
PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --granularity token \
    --target-tokens "$TARGET_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --shuffle \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy "${OUTPUT_DIR}"/documents.npy "${OUTPUT_DIR}"/doc_boundaries.npy
