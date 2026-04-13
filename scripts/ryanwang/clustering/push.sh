#!/bin/bash
# Push clustering outputs to S3 (exact mirror — deletes remote files not present locally).
#
# Excluded: embeddings_*.npy (large, regenerable from model checkpoints).
# Kept: assignments.npy, umap_coords.npy, documents.npy, doc_boundaries.npy (all small).
set -euo pipefail

aws s3 sync --delete \
    --exclude "*/embeddings_*.npy" \
    claude_outputs/clustering s3://ai2-sewonm/ryanwang/claude_outputs/clustering
