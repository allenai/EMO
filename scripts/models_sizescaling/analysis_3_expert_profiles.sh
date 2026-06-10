#!/bin/bash
# Phase 3: per-expert topic profiles + specialization-score distributions.
# Caches expert_profiles_<emb>.npz into each model's extraction dir.
# CPU-only, seconds. Requires analysis_1 extractions.
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_DIR="claude_outputs/models_sizescaling/weborganizer"

PYTHONUNBUFFERED=1 python -u -m src.scripts.clustering.expert_topic_profiles \
    --model-dirs \
        "${BASE_DIR}/emo_1b4b_130b" \
        "${BASE_DIR}/emo_1b7b_130b" \
        "${BASE_DIR}/emo_1b11b_130b" \
        "${BASE_DIR}/emo_1b14b_130b" \
    --labels 32e 64e 96e 128e \
    --output-dir claude_outputs/models_sizescaling/profiles
