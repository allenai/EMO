#!/bin/bash
# Phase 2: cross-model expert-usage trend curves (effective experts /
# coverage per topic vs total expert count). CPU-only, seconds.
# Requires analysis_1 extractions.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_DIR="claude_outputs/models_sizescaling/weborganizer"

PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.plot_expert_usage_trends \
    --model-dirs \
        "${BASE_DIR}/emo_1b4b_130b" \
        "${BASE_DIR}/emo_1b7b_130b" \
        "${BASE_DIR}/emo_1b11b_130b" \
        "${BASE_DIR}/emo_1b14b_130b" \
    --labels 32e 64e 96e 128e \
    --output-dir claude_outputs/models_sizescaling/trends
