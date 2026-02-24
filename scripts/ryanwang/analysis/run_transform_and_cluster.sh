#!/bin/bash
# Transform and cluster router embeddings.
#
# Usage:
#   # Sweep (no saving, just prints metrics):
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh logits mean_pca_l2 kmeans "8 16 32 64 128"
#
#   # Single run with saving:
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh logits mean_pca_l2 kmeans 64 --save
#
#   # List available options:
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh --list
set -e

DATA_DIR="claude_outputs/analysis/router_clustering_pretraining"

if [ "$1" = "--list" ]; then
    python -u -m src.scripts.analysis.transform_and_cluster --list
    exit 0
fi

EMBEDDING="${1:?Usage: $0 <embedding> <transform> <cluster> <k_values> [--save]}"
TRANSFORM="${2:?Usage: $0 <embedding> <transform> <cluster> <k_values> [--save]}"
CLUSTER="${3:?Usage: $0 <embedding> <transform> <cluster> <k_values> [--save]}"
K_VALUES="${4:?Usage: $0 <embedding> <transform> <cluster> <k_values> [--save]}"
SAVE_FLAG="${5:-}"

echo "=== Embedding: ${EMBEDDING}, Transform: ${TRANSFORM}, Cluster: ${CLUSTER}, k=${K_VALUES} ==="

OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
    --data-dir "$DATA_DIR" \
    --embedding "$EMBEDDING" \
    --transform "$TRANSFORM" \
    --cluster "$CLUSTER" \
    --k $K_VALUES \
    $SAVE_FLAG
