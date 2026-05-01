#!/bin/bash
# Cluster router embeddings.
#
# Usage:
#   bash scripts/clustering/cluster.sh <DATA_DIR> [EMBEDDING] [PREPROCESS] [METHOD] [K]
#
# Examples:
#   # Default: probs / mean_pca_l2 / spherical_kmeans / k=64, with save
#   bash scripts/clustering/cluster.sh claude_outputs/clustering/pretraining/<model>
#
#   # Custom config
#   bash scripts/clustering/cluster.sh claude_outputs/clustering/pretraining/<model> doc_topk_freq mean_pca_l2 spherical_kmeans 32
set -euo pipefail

DATA_DIR="${1:?Usage: $0 <DATA_DIR> [EMBEDDING] [PREPROCESS] [METHOD] [K]}"
EMBEDDING="${2:-probs}"
PREPROCESS="${3:-mean_pca_l2}"
METHOD="${4:-spherical_kmeans}"
K="${5:-64}"

echo "=== Clustering: ${EMBEDDING} / ${PREPROCESS} / ${METHOD} / k=${K} ==="
echo "=== Data: ${DATA_DIR} ==="

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.cluster \
    --data-dir "$DATA_DIR" \
    --embedding "$EMBEDDING" \
    --preprocess "$PREPROCESS" \
    --method "$METHOD" \
    --k $K \
    --save
