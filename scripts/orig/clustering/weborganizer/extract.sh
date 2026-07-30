#!/bin/bash
# Extract per-doc expert-coverage embeddings on the cc_all_dressed
# weborganizer topic mix.
#
# Standalone pipeline: produces embeddings_doc_topk_freq.npy and
# embeddings_doc_probs.npy directly from a single forward pass — does not
# go through clustering/extract.py + transform.py.
#
# Usage:
#   bash scripts/clustering/weborganizer/extract.sh <MODEL_PATH> [TARGET_TOKENS]
#
# MODEL_PATH may be a local checkpoint dir or an HF Hub id
# (e.g. allenai/Emo_1b14b_1T). HF Hub models load via trust_remote_code=True.
#
# Examples:
#   bash scripts/clustering/weborganizer/extract.sh \
#       models/<model>/step<N>-hf 20000000
#
#   # HF Hub id (set MODEL_NAME / BASE_DIR via env vars to control output)
#   MODEL_NAME=Emo_1b14b_1T BASE_DIR=cluster_eval_final/weborganizer \
#       bash scripts/clustering/weborganizer/extract.sh allenai/Emo_1b14b_1T
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/clustering/weborganizer/extract.sh ...
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <MODEL_PATH> [TARGET_TOKENS]}"
TARGET_TOKENS="${2:-20000000}"

# Output subdir name. Defaults to the local-path convention
# (basename(dirname(MODEL_PATH))). Override via the MODEL_NAME env var when
# MODEL_PATH is an HF Hub id, where dirname would give the org name.
MODEL_NAME="${MODEL_NAME:-$(basename "$(dirname "$MODEL_PATH")")}"
BASE_DIR="${BASE_DIR:-claude_outputs/clustering/weborganizer}"
COMPOSITION_FILE="${COMPOSITION_FILE:-${BASE_DIR}/mix_composition.json}"
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
