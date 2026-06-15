#!/usr/bin/env bash
# Grow each models_fullextend ghost checkpoint (50B / step11921) by one instantiated
# expert (128 -> 129) via uniform-average init, producing the `step11921-plus1`
# checkpoints that the continual-pretrain run (extend_finemath_frz.sh) loads.
#
# CPU-only weight surgery; idempotent (skips checkpoints already grown). Run once in a
# GPU-attached session before launching the extension runs.
#
#   bash scripts/models_fullextend/add_expert_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS_DIR="${MODELS_DIR:-models_fullextend}"
NUM_NEW_EXPERTS="${NUM_NEW_EXPERTS:-1}"

VARIANTS=(uniform usage random)

for v in "${VARIANTS[@]}"; do
    src="${MODELS_DIR}/emo_1b14b_130b_ghost_${v}_always_detachF/step11921"
    dst="${src}-plus${NUM_NEW_EXPERTS}"
    if [ -f "${dst}/config.json" ]; then
        echo "=== skip ${v}: ${dst} already exists ==="
        continue
    fi
    if [ ! -f "${src}/config.json" ]; then
        echo "!!! skip ${v}: source ${src} not found"
        continue
    fi
    echo "=== ${v}: ${src} -> ${dst} ==="
    python scripts/models_fullextend/add_expert_to_checkpoint.py \
        --checkpoint-path "${src}" \
        --save-path "${dst}" \
        --num-new-experts "${NUM_NEW_EXPERTS}"
done
