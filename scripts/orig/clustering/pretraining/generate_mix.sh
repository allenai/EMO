#!/bin/bash
# Generate pretraining data composition file.
# Queries S3 for file sizes and computes per-source token fractions.
# Only needs to be run once — the mix doesn't change.
#
# Output: ${OUTPUT_DIR}/pretraining_mix.json (default: claude_outputs/clustering/)
#
# Usage:
#   bash scripts/clustering/pretraining/generate_mix.sh
#   OUTPUT_DIR=cluster_eval_final bash scripts/clustering/pretraining/generate_mix.sh
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-claude_outputs/clustering}"
mkdir -p "$OUTPUT_DIR"

python -u -m src.scripts.clustering.generate_pretraining_mix \
    --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824.txt \
    --output "$OUTPUT_DIR/pretraining_mix.json" \
    --num-preview-docs 2 \
    --stream-bytes 3000000

echo ""
echo "=== Done. Output: ${OUTPUT_DIR}/pretraining_mix.json ==="
