#!/bin/bash
set -e

# ── Expert Coverage Analysis ────────────────────────────────────────────────
# Analyzes how MoE experts cover different weborganizer topic domains.
# Samples 20M tokens uniformly across topics and runs them through the model.
#
# Usage:
#   bash scripts/analysis/run_expert_coverage.sh [MODEL_PATH]
#
# If MODEL_PATH is not provided, defaults to the twolevelbatchlbreducedp model.
# Output dir is auto-derived from model name.
#
# Prerequisites:
#   - mix_composition.json from analyze_weborganizer.py (or will auto-generate)
#   - GPU available for model inference
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH="${1:-models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf}"
COMPOSITION_FILE="claude_outputs/analysis/router_clustering_weborganizer/mix_composition.json"
TARGET_TOKENS=20000000
BATCH_SIZE=32

# Use existing composition if available, otherwise let the script discover topics
COMP_ARG=""
if [ -f "$COMPOSITION_FILE" ]; then
    echo "Using existing composition: $COMPOSITION_FILE"
    COMP_ARG="--composition-file $COMPOSITION_FILE"
fi

# Output dir is auto-derived from model path by the python script
PYTHONUNBUFFERED=1 python -u \
    -m src.scripts.analysis.analyze_expert_coverage \
    --model-path "$MODEL_PATH" \
    $COMP_ARG \
    --target-tokens $TARGET_TOKENS \
    --batch-size $BATCH_SIZE \
    2>&1 | tee expert_coverage_$(basename "$(dirname "$MODEL_PATH")").log
