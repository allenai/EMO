#!/bin/bash
# Extract token-level router logits from the hellaswag_merged task
# (validation split), subsampled using the same seeded shuffle as the
# pruning calibration pipeline (src/hf_training/easy_ep_prune.py and
# greedy_prune_layerwise.py).
#
# Usage:
#   bash scripts/clustering/hellaswag/extract.sh [MODEL_PATH]
set -euo pipefail

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf}"

MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
OUTPUT_DIR="claude_outputs/clustering/hellaswag/${MODEL_NAME}"

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.clustering.extract \
    --source hellaswag \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-calibration 100 \
    --batch-size 32 \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy
