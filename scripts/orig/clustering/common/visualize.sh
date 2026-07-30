#!/bin/bash
# Generate interactive HTML cluster explorer.
#
# Usage:
#   bash scripts/clustering/common/visualize.sh <CLUSTER_DIR> [DATA_DIR]
#
# Example:
#   bash scripts/clustering/common/visualize.sh \
#       claude_outputs/clustering/pretraining/<model>/probs_mean_pca_l2_spherical_kmeans_k64
set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <CLUSTER_DIR> [DATA_DIR]}"
DATA_DIR="${2:-}"

ARGS="--cluster-dir $CLUSTER_DIR"
if [ -n "$DATA_DIR" ]; then
    ARGS="$ARGS --data-dir $DATA_DIR"
fi

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.visualize \
    $ARGS
