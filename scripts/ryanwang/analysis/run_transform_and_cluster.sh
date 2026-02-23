#!/bin/bash
# Transform and cluster router embeddings.
#
# Usage:
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh logits pca_l2
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh probs_sparse l2
#   bash scripts/ryanwang/analysis/run_transform_and_cluster.sh --list
set -e

DATA_DIR="claude_outputs/analysis/router_clustering_pretraining"

if [ "$1" = "--list" ]; then
    python -u -m src.scripts.analysis.transform_and_cluster --list
    exit 0
fi

EMBEDDING="${1:?Usage: $0 <embedding> <transform>}"
TRANSFORM="${2:?Usage: $0 <embedding> <transform>}"

echo "=== Embedding: ${EMBEDDING}, Transform: ${TRANSFORM} ==="

OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
    --data-dir "$DATA_DIR" \
    --embedding "$EMBEDDING" \
    --transform "$TRANSFORM"
