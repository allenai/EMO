#!/bin/bash
# Transform and cluster router embeddings.
#
# Usage:
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh logits mean_pca_l2 kmeans 64
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh probs_sparse l2 gmm 32
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh --list
set -e

DATA_DIR="claude_outputs/analysis/router_clustering_pretraining"

if [ "$1" = "--list" ]; then
    python -u -m src.scripts.analysis.transform_and_cluster --list
    exit 0
fi

EMBEDDING="${1:?Usage: $0 <embedding> <transform> <cluster> <k>}"
TRANSFORM="${2:?Usage: $0 <embedding> <transform> <cluster> <k>}"
CLUSTER="${3:?Usage: $0 <embedding> <transform> <cluster> <k>}"
K="${4:?Usage: $0 <embedding> <transform> <cluster> <k>}"

echo "=== Embedding: ${EMBEDDING}, Transform: ${TRANSFORM}, Cluster: ${CLUSTER}, k=${K} ==="

OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
    --data-dir "$DATA_DIR" \
    --embedding "$EMBEDDING" \
    --transform "$TRANSFORM" \
    --cluster "$CLUSTER" \
    --k "$K"
