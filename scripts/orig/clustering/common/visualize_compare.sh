#!/bin/bash
# Side-by-side comparison of two models' token-level clusterings as a
# single interactive HTML (Clusters tab + Documents tab).
#
# Both cluster dirs must come from extraction runs with the same shuffle
# seed & mix (i.e. identical documents.npy).
#
# Usage:
#   bash scripts/clustering/common/visualize_compare.sh \
#       <CLUSTER_DIR_1> <CLUSTER_DIR_2> [LABEL_1] [LABEL_2] [OUTPUT]
#
# Defaults:
#   LABEL_1   = "ModMoE"
#   LABEL_2   = "Standard MoE"
#   OUTPUT    = claude_outputs/clustering/pretraining/comparison.html
set -euo pipefail

CLUSTER_DIR_1="${1:?Usage: $0 <CLUSTER_DIR_1> <CLUSTER_DIR_2> [LABEL_1] [LABEL_2] [OUTPUT]}"
CLUSTER_DIR_2="${2:?Usage: $0 <CLUSTER_DIR_1> <CLUSTER_DIR_2> [LABEL_1] [LABEL_2] [OUTPUT]}"
LABEL_1="${3:-ModMoE}"
LABEL_2="${4:-Standard MoE}"
OUTPUT="${5:-claude_outputs/clustering/pretraining/comparison.html}"

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.visualize_compare \
    --cluster-dir-1 "$CLUSTER_DIR_1" \
    --cluster-dir-2 "$CLUSTER_DIR_2" \
    --label-1 "$LABEL_1" \
    --label-2 "$LABEL_2" \
    --sublabel-1 "(two-level MoE)" \
    --sublabel-2 "(baseline MoE)" \
    --subtitle-1 "learns topical / semantic clusters" \
    --subtitle-2 "learns syntactic / function-word clusters" \
    --output "$OUTPUT"

echo ""
echo "=== Done. Output: $OUTPUT ==="
ls -lh "$OUTPUT"
