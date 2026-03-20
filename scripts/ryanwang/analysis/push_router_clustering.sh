#!/bin/bash
# Push analysis outputs to S3, excluding large embedding files.
#
# Excluded: embeddings_*.npy and expert_freq.npy (100MB-4.6GB each).
# These can be regenerated from the model checkpoints.
# Kept: assignments.npy, umap_coords.npy, documents.npy, doc_boundaries.npy (all <5MB).
set -euo pipefail

aws s3 sync --delete \
    --exclude "*/embeddings_*.npy" \
    --exclude "*/expert_freq.npy" \
    claude_outputs/analysis s3://ai2-sewonm/ryanwang/claude_outputs/analysis
