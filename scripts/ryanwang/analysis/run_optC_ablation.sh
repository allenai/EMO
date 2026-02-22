#!/usr/bin/env bash
# optC/optD transform ablation — two-phase workflow:
#
# PHASE 1 (this script): sweep k ∈ {8,16,32,64,128} for all transforms.
#   Compare silhouette scores → pick the best embedding + transform.
#
# PHASE 2 (run_optC_phase2.sh): with the chosen transform, sweep k more
#   finely, then run final cluster + HTML viz.
#
# Assumes embeddings_optC_top32sparse.npy already exists in DATA_DIR.
# optD sweeps run only if embeddings_optD_logits_top32sparse.npy exists.

set -euo pipefail

export OPENBLAS_NUM_THREADS=16

DATA_DIR="claude_outputs/analysis/router_clustering_pretraining"
EMB_FILE_C="${DATA_DIR}/embeddings_optC_top32sparse.npy"
EMB_FILE_D="${DATA_DIR}/embeddings_optD_logits_top32sparse.npy"
BASE_OUT_C="${DATA_DIR}/optC_top32sparse"
BASE_OUT_D="${DATA_DIR}/optD_logits_top32sparse"

ALL_TRANSFORMS_C="raw l2 pca_l2 log_l2 standardize_pca_l2 renorm_l2 tsvd_l2 renorm_tsvd_l2"
ALL_TRANSFORMS_D="pca_l2 l2 tsvd_l2"

# --- optC sweeps ---
if [ ! -f "$EMB_FILE_C" ]; then
    echo "WARNING: $EMB_FILE_C not found. Skipping optC sweeps."
else
    for TRANSFORM in $ALL_TRANSFORMS_C; do
        OUT="${BASE_OUT_C}/${TRANSFORM}"
        mkdir -p "$OUT"

        # Skip if sweep already completed for this transform
        if [ -f "${OUT}/sweep.log" ] && grep -q "Sweep summary" "${OUT}/sweep.log"; then
            echo "Skipping optC/$TRANSFORM sweep (already done)"
            continue
        fi

        echo ""
        echo "========================================"
        echo "optC Transform: $TRANSFORM  [sweep]"
        echo "========================================"

        conda run --no-capture-output -n flexmoe python -u \
            -m src.scripts.analysis.cluster_embeddings \
            --output-dir "$OUT" \
            --emb-file "$EMB_FILE_C" \
            --data-dir "$DATA_DIR" \
            --mode sweep \
            --k-values 8 16 32 64 128 \
            --transform "$TRANSFORM" \
            2>&1 | tee "${OUT}/sweep.log"
    done
fi

# --- optD sweeps ---
if [ ! -f "$EMB_FILE_D" ]; then
    echo "WARNING: $EMB_FILE_D not found. Skipping optD sweeps."
    echo "  Run extraction with --embeddings optD first."
else
    for TRANSFORM in $ALL_TRANSFORMS_D; do
        OUT="${BASE_OUT_D}/${TRANSFORM}"
        mkdir -p "$OUT"

        # Skip if sweep already completed for this transform
        if [ -f "${OUT}/sweep.log" ] && grep -q "Sweep summary" "${OUT}/sweep.log"; then
            echo "Skipping optD/$TRANSFORM sweep (already done)"
            continue
        fi

        echo ""
        echo "========================================"
        echo "optD Transform: $TRANSFORM  [sweep]"
        echo "========================================"

        conda run --no-capture-output -n flexmoe python -u \
            -m src.scripts.analysis.cluster_embeddings \
            --output-dir "$OUT" \
            --emb-file "$EMB_FILE_D" \
            --data-dir "$DATA_DIR" \
            --mode sweep \
            --k-values 8 16 32 64 128 \
            --transform "$TRANSFORM" \
            2>&1 | tee "${OUT}/sweep.log"
    done
fi

# --- Summary ---
echo ""
echo "========================================"
echo "Phase 1 complete. Silhouette scores:"
echo ""

printf "  %-30s  %6s  %6s  %6s  %6s  %6s\n" "embedding/transform" "k=8" "k=16" "k=32" "k=64" "k=128"
printf "  %-30s  %6s  %6s  %6s  %6s  %6s\n" "------------------------------" "------" "------" "------" "------" "------"

if [ -f "$EMB_FILE_C" ]; then
    for TRANSFORM in $ALL_TRANSFORMS_C; do
        SILS=($(grep -oP 'silhouette=\K[0-9.-]+' "${BASE_OUT_C}/${TRANSFORM}/sweep.log" 2>/dev/null || echo "N/A N/A N/A N/A N/A"))
        printf "  %-30s  %6s  %6s  %6s  %6s  %6s\n" \
            "optC/$TRANSFORM" "${SILS[0]:-N/A}" "${SILS[1]:-N/A}" "${SILS[2]:-N/A}" "${SILS[3]:-N/A}" "${SILS[4]:-N/A}"
    done
fi

if [ -f "$EMB_FILE_D" ]; then
    for TRANSFORM in $ALL_TRANSFORMS_D; do
        SILS=($(grep -oP 'silhouette=\K[0-9.-]+' "${BASE_OUT_D}/${TRANSFORM}/sweep.log" 2>/dev/null || echo "N/A N/A N/A N/A N/A"))
        printf "  %-30s  %6s  %6s  %6s  %6s  %6s\n" \
            "optD/$TRANSFORM" "${SILS[0]:-N/A}" "${SILS[1]:-N/A}" "${SILS[2]:-N/A}" "${SILS[3]:-N/A}" "${SILS[4]:-N/A}"
    done
fi

echo ""
echo "Next: pick the best transform, then run:"
echo "  bash scripts/ryanwang/analysis/run_optC_phase2.sh <transform> <k>"
