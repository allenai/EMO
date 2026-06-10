#!/bin/bash
# Push the entire claude_outputs/ tree to S3, excluding large regeneratable files.
#
# S3 destination: s3://ai2-sewonm/ryanwang/emo_extend/claude_outputs/
# (the EMO-extension scratch tree; the original paper's scratch lives at
#  s3://ai2-sewonm/ryanwang/claude_outputs/ and is no longer synced here)
#
# Uses --delete so that files deleted locally are also removed on S3. Only run this
# from a machine whose local claude_outputs/ is at least a superset of the S3 copy —
# otherwise --delete will destroy S3 objects that exist only remotely. Pass the
# --dryrun flag to preview first.
#
# Excluded (regeneratable / too big to sync):
#   *.npy                 — embeddings, expert freq (up to multi-GB)
#   *.jsonl.gz            — metadata dumps
#   *.safetensors,*.bin,*.pt,*.pth — model checkpoints
#   *.arrow, *.parquet    — HF datasets
#   *.tar, *.tar.gz, *.zip — archives
#
# cluster_explorer.html (up to ~70MB each) is INCLUDED since it's a useful artifact.

set -euo pipefail

aws s3 sync --delete "$@" \
    --exclude "*.npy" \
    --exclude "*.jsonl.gz" \
    --exclude "*.safetensors" \
    --exclude "*.bin" \
    --exclude "*.pt" \
    --exclude "*.pth" \
    --exclude "*.arrow" \
    --exclude "*.parquet" \
    --exclude "*.tar" \
    --exclude "*.tar.gz" \
    --exclude "*.zip" \
    claude_outputs s3://ai2-sewonm/ryanwang/emo_extend/claude_outputs
