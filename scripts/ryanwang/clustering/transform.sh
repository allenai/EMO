#!/bin/bash
# Derive embeddings from raw logits.
#
# Usage:
#   bash scripts/ryanwang/clustering/transform.sh <DATA_DIR> <DERIVE>
#
# Examples:
#   bash scripts/ryanwang/clustering/transform.sh claude_outputs/clustering/pretraining/<model> probs
#   bash scripts/ryanwang/clustering/transform.sh claude_outputs/clustering/pretraining/<model> doc_topk_freq
#
# Available derivations: probs, topk_binary, doc_probs, doc_logits, doc_topk_freq
set -euo pipefail

DATA_DIR="${1:?Usage: $0 <DATA_DIR> <DERIVE>}"
DERIVE="${2:?Usage: $0 <DATA_DIR> <DERIVE>}"

echo "=== Deriving: ${DERIVE} from ${DATA_DIR} ==="

OPENBLAS_NUM_THREADS=16 python -u \
    -m src.scripts.clustering.transform \
    --data-dir "$DATA_DIR" \
    --derive "$DERIVE"
