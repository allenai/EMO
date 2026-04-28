#!/bin/bash
# Plot expert-coverage heatmaps from extract_document.py output.
#
# Usage:
#   bash scripts/ryanwang/clustering/weborganizer/plot.sh <DATA_DIR> [EMB_TYPE]
#
# DATA_DIR is the per-model dir produced by extract.sh
# (e.g. claude_outputs/clustering/weborganizer/<model_name>).
#
# EMB_TYPE one of {topk_freq, probs, both}. Default: both.
# When EMB_TYPE=both, runs the plotter twice and produces 10 PNGs.
set -euo pipefail

DATA_DIR="${1:?usage: $0 <DATA_DIR> [topk_freq|probs|both]}"
EMB_TYPE="${2:-both}"

# Shared topic-order file at the parent dir, so all models + embedding types
# render with the same row/column ordering. Created on first run, reused after.
TOPIC_ORDER_FILE="$(dirname "$DATA_DIR")/topic_order.json"

run_one() {
    local emb="$1"
    local emb_path="${DATA_DIR}/embeddings_doc_${emb}.npy"
    if [ ! -f "$emb_path" ]; then
        echo "ERROR: $emb_path not found"
        exit 1
    fi
    PYTHONUNBUFFERED=1 python -u \
        -m src.scripts.clustering.plot_doc_expert_coverage \
        --emb-file "$emb_path" \
        --topic-order-file "$TOPIC_ORDER_FILE" \
        2>&1 | tee "${DATA_DIR}/plot_${emb}.log"
}

case "$EMB_TYPE" in
    topk_freq) run_one topk_freq ;;
    probs)     run_one probs ;;
    both)      run_one topk_freq; run_one probs ;;
    *) echo "ERROR: unknown EMB_TYPE '$EMB_TYPE' (want topk_freq|probs|both)"; exit 1 ;;
esac

echo ""
echo "=== Done. Plots in ${DATA_DIR}/ ==="
ls -lh "${DATA_DIR}"/*.png
