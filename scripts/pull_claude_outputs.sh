#!/bin/bash
# Pull the claude_outputs/ tree from S3 to local.
#
# Intentionally does NOT use --delete: the local machine (e.g. a laptop) may have
# extra artifacts that aren't on S3, and we don't want to wipe them. Pass --dryrun
# to preview.

set -euo pipefail

aws s3 sync "$@" \
    s3://ai2-sewonm/ryanwang/claude_outputs claude_outputs
