#!/bin/bash
# Extract token-level router logits from pretraining data.
#
# Supports multi-GPU: set CUDA_VISIBLE_DEVICES to control which GPUs to use.
# The model is loaded with device_map="auto" which shards across available GPUs.
#
# Usage:
#   bash scripts/clustering/pretraining/extract.sh <MODEL_PATH> [TARGET_TOKENS] [MAX_TOKENS_PER_DOC]
#
# MODEL_PATH may be a local checkpoint dir or an HF Hub id
# (e.g. allenai/Emo_1b14b_1T). HF Hub models load via trust_remote_code=True.
#
# Examples:
#   # Local checkpoint (output subdir defaults to the local run name)
#   bash scripts/clustering/pretraining/extract.sh models/moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf
#
#   # HF Hub id (set MODEL_NAME to control the output subdir name; otherwise
#   # the basename(dirname()) derivation gives e.g. "allenai", which collides
#   # across models from the same org)
#   MODEL_NAME=Emo_1b14b_1T bash scripts/clustering/pretraining/extract.sh allenai/Emo_1b14b_1T
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/clustering/pretraining/extract.sh <MODEL_PATH>
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <MODEL_PATH> [TARGET_TOKENS] [MAX_TOKENS_PER_DOC]}"
TARGET_TOKENS="${2:-1000000}"
MAX_TOKENS_PER_DOC="${3:-100}"

# Output subdir name. Defaults to the local-path convention
# (basename(dirname(MODEL_PATH))). Override via the MODEL_NAME env var when
# MODEL_PATH is an HF Hub id, where dirname would give the org name (e.g.
# "allenai") and collide across different models from the same org.
MODEL_NAME="${MODEL_NAME:-$(basename "$(dirname "$MODEL_PATH")")}"
BASE_DIR="${BASE_DIR:-claude_outputs/clustering/pretraining}"
COMPOSITION_FILE="${COMPOSITION_FILE:-claude_outputs/clustering/pretraining_mix.json}"
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}"

echo "Model: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Target tokens: $TARGET_TOKENS (post-truncation)"
echo "Max tokens/doc: $MAX_TOKENS_PER_DOC"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

if [ ! -f "$COMPOSITION_FILE" ]; then
    echo "ERROR: ${COMPOSITION_FILE} not found."
    echo "Run: bash scripts/clustering/pretraining/generate_mix.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.clustering.extract \
    --source pretraining \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens "$TARGET_TOKENS" \
    --max-tokens-per-doc "$MAX_TOKENS_PER_DOC" \
    --batch-size 32 \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy "${OUTPUT_DIR}"/documents.npy "${OUTPUT_DIR}"/doc_boundaries.npy
