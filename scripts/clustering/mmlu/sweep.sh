#!/bin/bash
# MMLU-specific clustering sweep.
#
# Fixes k=16 (rough target = ~1 cluster per ~3-4 subjects at 57-subject granularity)
# and iterates the knobs actually worth exploring for "find better subject
# groupings than the manual 17 categories":
#
#   embedding   : doc_probs, doc_topk_freq, doc_layer0_probs
#   balance     : off, source   (source-balanced to min-class count, 68/subject)
#   method      : kmeans, spherical_kmeans
#   preprocess  : mean_pca_l2   (fixed)
#
# Total: 3 × 2 × 2 = 12 clusterings. Each is seconds on 9,937-row doc-level
# data, so the whole sweep is minutes. Logs are tee'd per-config into the
# data dir.
#
# Usage:
#   bash scripts/clustering/mmlu/sweep.sh <MMLU_DATA_DIR>
#
# Example:
#   bash scripts/clustering/mmlu/sweep.sh \
#       claude_outputs/clustering/mmlu/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301
set -uo pipefail

DATA_DIR="${1:?Usage: $0 <MMLU_DATA_DIR>}"

K=16
PREPROCESS="mean_pca_l2"
EMBEDDINGS=(doc_probs doc_topk_freq doc_layer0_probs doc_layer15_probs)
BALANCES=(off source)                 # "off" => unbalanced; anything else => --balance-by <val>
METHODS=(kmeans spherical_kmeans)

echo "============================================================"
echo "  MMLU sweep (k=${K}, preprocess=${PREPROCESS})"
echo "  Data: ${DATA_DIR}"
echo "  Embeddings: ${EMBEDDINGS[*]}"
echo "  Balances:   ${BALANCES[*]}"
echo "  Methods:    ${METHODS[*]}"
echo "============================================================"

for EMB in "${EMBEDDINGS[@]}"; do
    # Auto-derive the embedding if not already present.
    EMB_FILE="${DATA_DIR}/embeddings_${EMB}.npy"
    if [ ! -f "$EMB_FILE" ]; then
        echo ""
        echo ">>> Deriving ${EMB} ..."
        bash scripts/clustering/common/transform.sh "$DATA_DIR" "$EMB" \
            2>&1 | tee "${DATA_DIR}/derive_${EMB}.log"
    fi

    for BAL in "${BALANCES[@]}"; do
        if [ "$BAL" = "off" ]; then
            BAL_ARGS=()
            BAL_TAG="balOFF"
        else
            BAL_ARGS=(--balance-by "$BAL")
            BAL_TAG="bal${BAL}"
        fi

        for METHOD in "${METHODS[@]}"; do
            SWEEP_LOG="${DATA_DIR}/sweep_${EMB}_${PREPROCESS}_${METHOD}_k${K}_${BAL_TAG}.log"
            echo ""
            echo "============================================================"
            echo "  ${EMB} / ${PREPROCESS} / ${METHOD} / k=${K} / balance=${BAL}"
            echo "  -> ${SWEEP_LOG}"
            echo "============================================================"

            OPENBLAS_NUM_THREADS=16 python -u \
                -m src.scripts.clustering.cluster \
                --data-dir "$DATA_DIR" \
                --embedding "$EMB" \
                --preprocess "$PREPROCESS" \
                --method "$METHOD" \
                --k "$K" \
                "${BAL_ARGS[@]}" \
                2>&1 | tee "$SWEEP_LOG" \
                || echo "!!! FAILED: ${EMB} / ${METHOD} / balance=${BAL} !!!"
        done
    done
done

echo ""
echo "=== MMLU sweep complete ==="
