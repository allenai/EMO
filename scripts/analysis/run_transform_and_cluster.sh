#!/bin/bash
# Transform and cluster router embeddings.
#
# Usage:
#   # Sweep (no saving, just prints metrics):
#   bash scripts/analysis/run_transform_and_cluster.sh <DATA_DIR> logits mean_pca_l2 kmeans "8 16 32 64 128"
#
#   # Single run with saving:
#   bash scripts/analysis/run_transform_and_cluster.sh <DATA_DIR> logits mean_pca_l2 kmeans 64 --save
#
#   # List available options:
#   bash scripts/analysis/run_transform_and_cluster.sh --list
set -e

if [ "$1" = "--list" ]; then
    python -u -m src.scripts.analysis.transform_and_cluster --list
    exit 0
fi

DATA_DIR="${1:?Usage: $0 <data_dir> <embedding> <transform> <cluster> <k_values> [--save]}"
EMBEDDING="${2:?Usage: $0 <data_dir> <embedding> <transform> <cluster> <k_values> [--save]}"
TRANSFORM="${3:?Usage: $0 <data_dir> <embedding> <transform> <cluster> <k_values> [--save]}"
CLUSTER="${4:?Usage: $0 <data_dir> <embedding> <transform> <cluster> <k_values> [--save]}"
K_VALUES="${5:?Usage: $0 <data_dir> <embedding> <transform> <cluster> <k_values> [--save]}"
SAVE_FLAG="${6:-}"

echo "=== Data: ${DATA_DIR} ==="
echo "=== Embedding: ${EMBEDDING}, Transform: ${TRANSFORM}, Cluster: ${CLUSTER}, k=${K_VALUES} ==="

OPENBLAS_NUM_THREADS=16 python -u -m src.scripts.analysis.transform_and_cluster \
    --data-dir "$DATA_DIR" \
    --embedding "$EMBEDDING" \
    --transform "$TRANSFORM" \
    --cluster "$CLUSTER" \
    --k $K_VALUES \
    $SAVE_FLAG
