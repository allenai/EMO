#!/bin/bash
# Rich 4-view token-level cluster explorer (Clusters / Documents / Tokens / UMAP).
#
# Usage:
#   bash scripts/ryanwang/clustering/common/visualize_token.sh <CLUSTER_DIR> [CONTEXT_WINDOW]
#
# Example:
#   bash scripts/ryanwang/clustering/common/visualize_token.sh \
#       claude_outputs/clustering/pretraining/<model>/probs_mean_pca_l2_spherical_kmeans_k32
#
# Reads cluster_labels.json from CLUSTER_DIR if present.
# Writes cluster_explorer.html (overwriting any simple visualizer output).
set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <CLUSTER_DIR> [CONTEXT_WINDOW]}"
CONTEXT_WINDOW="${2:-10}"

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.visualize_token \
    --cluster-dir "$CLUSTER_DIR" \
    --context-window "$CONTEXT_WINDOW"
