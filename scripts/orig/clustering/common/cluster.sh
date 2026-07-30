#!/bin/bash
# Cluster router embeddings.
#
# Usage:
#   bash scripts/clustering/common/cluster.sh <DATA_DIR> [EMBEDDING] [PREPROCESS] [METHOD] [K] [BALANCE_BY] [BALANCE_N]
#
# Examples:
#   # Default: probs / mean_pca_l2 / spherical_kmeans / k=64, with save
#   bash scripts/clustering/common/cluster.sh claude_outputs/clustering/pretraining/<model>
#
#   # Custom config
#   bash scripts/clustering/common/cluster.sh claude_outputs/clustering/pretraining/<model> doc_topk_freq mean_pca_l2 spherical_kmeans 32
#
#   # Balanced by source (stratified subsample to min class count)
#   bash scripts/clustering/common/cluster.sh <DIR> doc_probs mean_pca_l2 spherical_kmeans 32 source
#
#   # Balanced by source, capped at 100 per class
#   bash scripts/clustering/common/cluster.sh <DIR> doc_probs mean_pca_l2 spherical_kmeans 32 source 100
set -euo pipefail

DATA_DIR="${1:?Usage: $0 <DATA_DIR> [EMBEDDING] [PREPROCESS] [METHOD] [K] [BALANCE_BY] [BALANCE_N]}"
EMBEDDING="${2:-probs}"
PREPROCESS="${3:-mean_pca_l2}"
METHOD="${4:-spherical_kmeans}"
K="${5:-64}"
BALANCE_BY="${6:-}"
BALANCE_N="${7:-}"

BALANCE_ARGS=()
if [ -n "$BALANCE_BY" ]; then
    BALANCE_ARGS+=(--balance-by "$BALANCE_BY")
    if [ -n "$BALANCE_N" ]; then
        BALANCE_ARGS+=(--balance-n "$BALANCE_N")
    fi
fi

echo "=== Clustering: ${EMBEDDING} / ${PREPROCESS} / ${METHOD} / k=${K} ==="
echo "=== Data: ${DATA_DIR} ==="
if [ -n "$BALANCE_BY" ]; then
    echo "=== Balance: by=${BALANCE_BY} n=${BALANCE_N:-min} ==="
fi

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.cluster \
    --data-dir "$DATA_DIR" \
    --embedding "$EMBEDDING" \
    --preprocess "$PREPROCESS" \
    --method "$METHOD" \
    --k $K \
    --save \
    "${BALANCE_ARGS[@]}"
