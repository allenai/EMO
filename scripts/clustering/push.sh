#!/bin/bash
# Push clustering outputs to S3 (exact mirror — deletes remote files not present locally).
#
# Only pushes visualization and metadata files:
#   - cluster_explorer.html, cluster_labels.json
#   - run_info.json, summary.json, info.json
#   - metadata_*.jsonl.gz
#   - pretraining_mix.json
#
# Excludes all .npy files (large, regenerable).
set -euo pipefail

aws s3 sync --delete \
    --exclude "*" \
    --include "*.html" \
    --include "*.json" \
    --include "*.jsonl.gz" \
    --include "*.log" \
    claude_outputs/clustering s3://ai2-sewonm/ryanwang/claude_outputs/clustering
