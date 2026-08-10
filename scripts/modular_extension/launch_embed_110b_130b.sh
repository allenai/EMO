#!/usr/bin/env bash
# Full embed sweep of the 110B-130B doc window (see run_extract_110b_130b.sh) with the
# EMO 100B checkpoint: all 128 shards. The window is 2x the 100B-110B one (~20B tokens),
# so 8 Beaker jobs x 8 GPUs to keep wall-clock similar to the first window's sweep.
# Embeddings land in a sibling dir of the first window's, keyed by window; clustering
# can then merge both windows' shards.
#
# Run only AFTER the extraction completes (workers glob docs-*.jsonl.gz at startup; a
# partial extraction would silently embed a subset). Guarded by a manifest check.
#
#   bash scripts/modular_extension/launch_embed_110b_130b.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
DATA_DIR="${WEKA_ROOT}/modular_extension/data/emo_64exp_50b_wsd_lr2e-3_110B-130B"
# The /weka path only resolves inside Beaker workers; locally the same storage is the
# repo root (see CLAUDE.md), so the completeness guard checks the local twin.
LOCAL_DATA_DIR="$(cd ../.. && pwd)/modular_extension/data/emo_64exp_50b_wsd_lr2e-3_110B-130B"
[ -f "${LOCAL_DATA_DIR}/manifest.json" ] || {
    echo "ERROR: ${LOCAL_DATA_DIR}/manifest.json missing -- extraction not finished" >&2
    exit 1
}

DOCS_GLOB="${DATA_DIR}/docs-*.jsonl.gz" \
OUTPUT_DIR="${WEKA_ROOT}/modular_extension/cluster/emo100b_step23842/embeddings_110B-130B" \
JOB_PREFIX="modext-embed-docs-110b130b" \
SHARDS="$(seq -s, 0 127)" JOBS=8 bash launch_embed_docs.sh
