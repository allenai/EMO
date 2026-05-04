#!/bin/bash
# Extract per-doc expert-coverage embeddings on the cc_all_dressed
# weborganizer topic mix.
#
# Standalone pipeline: produces embeddings_doc_topk_freq.npy and
# embeddings_doc_probs.npy directly from a single forward pass — does not
# go through clustering/extract.py + transform.py.
#
# Usage:
#   bash scripts/clustering/weborganizer/extract.sh [MODEL_PATH] [TARGET_TOKENS]
#
# Examples:
#   bash scripts/clustering/weborganizer/extract.sh
#   bash scripts/clustering/weborganizer/extract.sh \
#       models/<model>/step<N>-hf 20000000
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/clustering/weborganizer/extract.sh ...
set -euo pipefail

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf}"
TARGET_TOKENS="${2:-20000000}"

MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
BASE_DIR="claude_outputs/clustering/weborganizer"
COMPOSITION_FILE="${BASE_DIR}/mix_composition.json"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"

echo "Model:         $MODEL_PATH"
echo "Model name:    $MODEL_NAME"
echo "Output dir:    $OUTPUT_DIR"
echo "Composition:   $COMPOSITION_FILE  (auto-generated if missing)"
echo "Target tokens: $TARGET_TOKENS"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-all}"

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.clustering.extract_document \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --composition-file "$COMPOSITION_FILE" \
    --target-tokens "$TARGET_TOKENS" \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_doc_*.npy "${OUTPUT_DIR}"/info.json
