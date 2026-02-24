#!/bin/bash
set -e

# Temporary test script for debugging analyze_expert_coverage.py with pdb.
# Uses minimal tokens and no output piping so pdb works interactively.

MODEL_PATH="models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
COMPOSITION_FILE="claude_outputs/analysis/router_clustering_weborganizer/mix_composition.json"
OUTPUT_DIR="claude_outputs/analysis/expert_coverage_test"

mkdir -p "$OUTPUT_DIR"

COMP_ARG=""
if [ -f "$COMPOSITION_FILE" ]; then
    echo "Using existing composition: $COMPOSITION_FILE"
    COMP_ARG="--composition-file $COMPOSITION_FILE"
fi

python -m src.scripts.analysis.analyze_expert_coverage \
    --model-path "$MODEL_PATH" \
    $COMP_ARG \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens 50000 \
    --batch-size 4 \
    --debug
