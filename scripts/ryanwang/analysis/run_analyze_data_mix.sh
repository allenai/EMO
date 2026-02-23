#!/bin/bash
# Analyze data composition for pretraining mix.
# Outputs mix_composition.json with per-source token fractions.
# Only needs to be run once per data mix.
set -e

OUTPUT_DIR="claude_outputs/analysis/router_clustering_pretraining"

python -u -m src.scripts.analysis.analyze_data_mix \
    --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824.txt \
    --output-dir "$OUTPUT_DIR" \
    --num-preview-docs 2 \
    --stream-bytes 3000000

echo ""
echo "=== Done. Output: ${OUTPUT_DIR}/mix_composition.json ==="
