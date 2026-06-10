#!/bin/bash
# Phase 1 of the sizescaling specialization analysis: weborganizer doc-level
# expert-coverage extraction + stock heatmap plots for all four
# models_sizescaling checkpoints (32/64/96/128 experts).
#
# Wraps the generic scripts/clustering/weborganizer/{extract,plot}.sh.
# Idempotent: skips models whose embeddings already exist.
#
# All four extractions share mix_composition.json (same docs, shuffle seed 42),
# so per-document comparisons across models are valid. Plots share
# topic_order.json at the parent dir, so heatmaps are row-aligned across
# models (and with the earlier 1T-anneal runs pulled from S3).
#
# Usage:
#   bash scripts/models_sizescaling/analysis_1_weborganizer_extract.sh
#
# Runtime: ~40 min/model on one A100-80GB (33 min forward + S3 doc loading).
# Output:  claude_outputs/clustering/weborganizer/<run_name>/
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=(emo_1b4b_130b emo_1b7b_130b emo_1b11b_130b emo_1b14b_130b)
TARGET_TOKENS="${TARGET_TOKENS:-20000000}"
STEP="${STEP:-step30995}"
BASE_DIR="claude_outputs/clustering/weborganizer"

for m in "${MODELS[@]}"; do
    ckpt="models_sizescaling/${m}/${STEP}-hf"
    out="${BASE_DIR}/${m}"
    if [ -f "${out}/embeddings_doc_probs.npy" ]; then
        echo "=== ${m}: extraction exists, skipping ==="
    else
        echo "=== ${m}: extracting (${ckpt}) ==="
        bash scripts/clustering/weborganizer/extract.sh "${ckpt}" "${TARGET_TOKENS}"
    fi
    echo "=== ${m}: plotting ==="
    bash scripts/clustering/weborganizer/plot.sh "${out}" both
done

echo ""
echo "=== All done. Per-model outputs under ${BASE_DIR}/ ==="
