#!/bin/bash
# Extract token-level router logits from the 57 per-subject MMLU tasks
# mmlu_merged_<subject>:rc_validation::olmes — the merged variant uses
# test[:60%]+validation shuffled (seed=0) per subject. All prompts are
# used; no subsampling. Each prompt is wrapped in a subject-matched
# 5-shot OLMES RC context.
#
# Usage:
#   bash scripts/ryanwang/clustering/mmlu/extract.sh [MODEL_PATH]
set -euo pipefail

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf}"

MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")"
OUTPUT_DIR="claude_outputs/clustering/mmlu/${MODEL_NAME}"

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.clustering.extract \
    --source mmlu \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 32 \
    2>&1 | tee "${OUTPUT_DIR}/extraction.log"

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR}/ ==="
ls -lh "${OUTPUT_DIR}"/embeddings_*.npy
