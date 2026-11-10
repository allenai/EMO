#!/usr/bin/env bash
# PARENT: scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh
# DESCRIPTION:
#     debug_validation arm (user request 2026-11-10): the OLMoE3-ladder 275M EMO 512e model (10B tokens, WSD-trunk recipe,
#     4 jupiter nodes x 8 H100, allocated) with a MIXED pool selection: for a random 20% of the training documents the
#     expert pool is a uniformly random set of d experts (as olmoe3_275m_emo_randsel_10b), for the other 80% it is EMO's
#     relevance pool (the d experts with the largest summed router score over the document). The pool size d is drawn as
#     always (uniform in [16, 512]); eval pool 512, eval-time selection unchanged. Implemented by
#     OLMOE3_EMO_POOL_SELECT=mix:0.2 in scripts/sparse_experts/olmoe3_275m.py (per-document coin, in-process router patch).
#
#     Checkpoints: /weka/oe-training-default/ryanwang/EMO/debug_validation/olmoe3_275m_emo_mix20_10b (permanent every
#     5000 steps + final); in-loop v3-small ppl validation every 1000 steps; W&B project emo-extension, tags
#     [pretraining, debug_validation, olmoe3_275m, 512e, emo, pool_select_mix0.2, pool_min16, jupiter, 10b].
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/debug_validation/model_scripts/olmoe3_275m_emo_mix20_10b.sh [dry_run]
##############################################################
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=mix:0.2
export OLMOE3_EMO_MIN_POOL="${OLMOE3_EMO_MIN_POOL:-16}"
export OLMOE3_EXPERIMENT=debug_validation
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_mix20_10b}"
export OLMOE3_SAVE_ROOT="${OLMOE3_SAVE_ROOT:-/weka/oe-training-default/ryanwang/EMO/debug_validation}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-10b}"
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_PREEMPTIBLE=0   # allocated
export OLMOE3_PPL_EVAL_INTERVAL="${OLMOE3_PPL_EVAL_INTERVAL:-1000}"   # in-loop v3-small ppl validation (11 sets) every 1000 steps + at the end
export OLMOE3_FOLLOW="${OLMOE3_FOLLOW:-0}"   # submit and return; watch with beaker CLI
source "$(dirname "${BASH_SOURCE[0]}")/../../sparse_experts/model_scripts/olmoe3_275m_10b.sh" "$@"
