#!/bin/bash
# Sweep clustering configurations.
#
# Usage:
#   bash scripts/ryanwang/clustering/sweep.sh <DATA_DIR> [EMBEDDING]
#
# Default grid: {probs} × {mean_pca_l2} × {kmeans, spherical_kmeans} × k={16,32,64,128}
set -uo pipefail

DATA_DIR="${1:?Usage: $0 <DATA_DIR> [EMBEDDING]}"
EMBEDDING="${2:-probs}"

PREPROCESS="mean_pca_l2"
METHODS="kmeans spherical_kmeans"
K_VALUES="16 32 64 128"

echo "=== Sweep: ${EMBEDDING} / ${PREPROCESS} ==="
echo "=== Methods: ${METHODS} ==="
echo "=== k values: ${K_VALUES} ==="
echo "=== Data: ${DATA_DIR} ==="

for METHOD in $METHODS; do
    echo ""
    echo "================================================================"
    echo "  ${EMBEDDING} / ${PREPROCESS} / ${METHOD}"
    echo "================================================================"

    OPENBLAS_NUM_THREADS=16 python -u \
        -m src.scripts.clustering.cluster \
        --data-dir "$DATA_DIR" \
        --embedding "$EMBEDDING" \
        --preprocess "$PREPROCESS" \
        --method "$METHOD" \
        --k $K_VALUES \
        || echo "!!! FAILED: ${EMBEDDING} / ${PREPROCESS} / ${METHOD} !!!"
done

echo ""
echo "=== Sweep complete ==="
