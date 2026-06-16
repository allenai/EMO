#!/usr/bin/env bash
# Grow each models_fullextend 50B / step11921 checkpoint by one instantiated expert
# (128 -> 129) via uniform-average init, producing the `step11921-plus1` checkpoints that
# the continual-pretrain run (extend_finemath_frz_*.sh) loads. Covers the three ghost
# variants AND the no-ghost EMO baseline (the apples-to-apples control).
#
# CPU-only weight surgery; idempotent (skips checkpoints already grown). Run once in a
# GPU-attached session before launching the extension runs.
#
#   bash scripts/models_fullextend/add_expert_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS_DIR="${MODELS_DIR:-models_fullextend}"
NUM_NEW_EXPERTS="${NUM_NEW_EXPERTS:-1}"

# Source run-dir names of the 50B checkpoints to grow.
RUNS=(
    emo_1b14b_130b_ghost_uniform_always_detachF
    emo_1b14b_130b_ghost_usage_always_detachF
    emo_1b14b_130b_ghost_random_always_detachF
    emo_1b14b_130b   # no-ghost EMO baseline (control)
)

for run in "${RUNS[@]}"; do
    src="${MODELS_DIR}/${run}/step11921"
    dst="${src}-plus${NUM_NEW_EXPERTS}"
    if [ -f "${dst}/config.json" ]; then
        echo "=== skip ${run}: ${dst} already exists ==="
        continue
    fi
    if [ ! -f "${src}/config.json" ]; then
        echo "!!! skip ${run}: source ${src} not found"
        continue
    fi
    echo "=== ${run}: ${src} -> ${dst} ==="
    python scripts/models_fullextend/add_expert_to_checkpoint.py \
        --checkpoint-path "${src}" \
        --save-path "${dst}" \
        --num-new-experts "${NUM_NEW_EXPERTS}"
done
