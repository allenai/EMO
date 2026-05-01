#!/bin/bash
# Derive probs+topk_binary for two 1T models and run 4 sweeps
# (probs/topk_binary × model1/model2) with mean_pca_l2 preprocessing.
set -uo pipefail

BASE="claude_outputs/clustering/pretraining"
MODELS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419"
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419"
)
EMBEDDINGS=(probs topk_binary)

for MODEL in "${MODELS[@]}"; do
    DATA_DIR="${BASE}/${MODEL}"
    echo "================================================================"
    echo "  Model: ${MODEL}"
    echo "================================================================"

    for EMB in "${EMBEDDINGS[@]}"; do
        EMB_FILE="${DATA_DIR}/embeddings_${EMB}.npy"
        if [ ! -f "$EMB_FILE" ]; then
            echo ">>> Deriving ${EMB} for ${MODEL} ..."
            OPENBLAS_NUM_THREADS=16 python -u \
                -m src.scripts.clustering.transform \
                --data-dir "$DATA_DIR" \
                --derive "$EMB" \
                2>&1 | tee "${DATA_DIR}/derive_${EMB}.log"
        else
            echo ">>> ${EMB_FILE} already exists, skipping derive."
        fi
    done

    for EMB in "${EMBEDDINGS[@]}"; do
        SWEEP_LOG="${DATA_DIR}/${EMB}_mean_pca_l2_sweep.log"
        echo ">>> Sweeping ${EMB} (mean_pca_l2) ..."
        bash scripts/clustering/sweep.sh "$DATA_DIR" "$EMB" \
            2>&1 | tee "$SWEEP_LOG"
    done
done

echo ""
echo "=== All 4 sweeps complete ==="
