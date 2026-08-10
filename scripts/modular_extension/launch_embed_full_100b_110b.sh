#!/usr/bin/env bash
# Full embed sweep of the 100B-110B doc window: all 128 shards, 4 Beaker jobs x 8 GPUs.
# Shard outputs are idempotent, so the already-computed pilot shards 0-15 are skipped and
# only the remaining 112 shards cost GPU time.
#
#   bash scripts/modular_extension/launch_embed_full_100b_110b.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SHARDS="$(seq -s, 0 127)" JOBS=4 bash launch_embed_docs.sh
