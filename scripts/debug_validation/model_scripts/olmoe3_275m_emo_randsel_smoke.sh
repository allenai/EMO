#!/usr/bin/env bash
# 1-GPU, 40-step smoke of the random-pool-selection arm (checks the in-process router patch trains
# on Beaker end to end). Saves under debug_validation/smoke; no in-loop ppl eval.
#   bash scripts/debug_validation/model_scripts/olmoe3_275m_emo_randsel_smoke.sh [dry_run]
export OLMOE3_TOKENS=$((40 * 524288))
export OLMOE3_RUNNAME=olmoe3_275m_emo_randsel_smoke
export OLMOE3_WANDB_TAGS=smoke
export OLMOE3_SAVE_ROOT=/weka/oe-training-default/ryanwang/EMO/debug_validation/smoke
export OLMOE3_NUM_NODES=1
export OLMOE3_NUM_GPUS=1
export OLMOE3_PPL_EVAL_INTERVAL=0   # no in-loop ppl eval in the smoke
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_emo_randsel_10b.sh" "${1:-launch}" \
    --trainer.callbacks.checkpointer.save_interval=20 \
    --trainer.callbacks.checkpointer.ephemeral_save_interval=null
