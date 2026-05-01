#!/bin/bash
# Pull clustering outputs from S3 (additive — does not delete local files).
set -euo pipefail

aws s3 sync \
    s3://ai2-sewonm/ryanwang/claude_outputs/clustering claude_outputs/clustering
