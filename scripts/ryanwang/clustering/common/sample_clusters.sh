#!/bin/bash
# Sample token contexts per cluster for manual labeling.
#
# Usage:
#   bash scripts/ryanwang/clustering/common/sample_clusters.sh <CLUSTER_DIR> [N_SAMPLES]
#
# Example:
#   bash scripts/ryanwang/clustering/common/sample_clusters.sh \
#       claude_outputs/clustering/pretraining/<model>/probs_mean_pca_l2_spherical_kmeans_k32
#
# Writes cluster_samples/cluster_NN.txt files into CLUSTER_DIR.
set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <CLUSTER_DIR> [N_SAMPLES]}"
N_SAMPLES="${2:-200}"

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.sample_clusters \
    --cluster-dir "$CLUSTER_DIR" \
    --n-samples "$N_SAMPLES"
