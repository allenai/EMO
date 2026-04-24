#!/bin/bash
# Sweep clustering configurations.
#
# Usage:
#   bash scripts/ryanwang/clustering/sweep.sh <DATA_DIR> [EMBEDDING] [BALANCE_BY] [BALANCE_N]
#
# Default grid: {probs} × {mean_pca_l2} × {kmeans, spherical_kmeans} × k={16,32,64,128}
set -uo pipefail

DATA_DIR="${1:?Usage: $0 <DATA_DIR> [EMBEDDING] [BALANCE_BY] [BALANCE_N]}"
EMBEDDING="${2:-probs}"
BALANCE_BY="${3:-}"
BALANCE_N="${4:-}"

PREPROCESS="mean_pca_l2"
METHODS="kmeans spherical_kmeans"
K_VALUES="16 32 64 128"

BALANCE_ARGS=()
if [ -n "$BALANCE_BY" ]; then
    BALANCE_ARGS+=(--balance-by "$BALANCE_BY")
    if [ -n "$BALANCE_N" ]; then
        BALANCE_ARGS+=(--balance-n "$BALANCE_N")
    fi
fi

echo "=== Sweep: ${EMBEDDING} / ${PREPROCESS} ==="
echo "=== Methods: ${METHODS} ==="
echo "=== k values: ${K_VALUES} ==="
echo "=== Data: ${DATA_DIR} ==="
if [ -n "$BALANCE_BY" ]; then
    echo "=== Balance: by=${BALANCE_BY} n=${BALANCE_N:-min} ==="
fi

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
        "${BALANCE_ARGS[@]}" \
        || echo "!!! FAILED: ${EMBEDDING} / ${PREPROCESS} / ${METHOD} !!!"
done

echo ""
echo "=== Sweep complete ==="
