#!/bin/bash
set -e

# Temporary test script for debugging extract_router_embeddings.py.
# Uses --debug (2 sources only), minimal tokens, small batch size,
# and only the topk_freq embedding for fast iteration.
#
# Usage:
#   bash scripts/ryanwang/analysis/test_extract_embeddings.sh

MODEL_PATH="models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
COMPOSITION_FILE="claude_outputs/analysis/router_clustering_pretraining/mix_composition.json"
OUTPUT_DIR="claude_outputs/analysis/extract_embeddings_test"

mkdir -p "$OUTPUT_DIR"

python -u -m src.scripts.analysis.extract_router_embeddings \
    --model-path "$MODEL_PATH" \
    --composition-file "$COMPOSITION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --target-tokens 50000 \
    --batch-size 4 \
    --embeddings topk_freq \
    --debug
