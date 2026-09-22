#!/usr/bin/env bash
# PARENT: scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh
# DESCRIPTION:
#     debug_validation arm (user request 2026-09-22): the OLMoE3-ladder 275M EMO model (512 routed
#     experts + 1 shared, top-16, identical architecture, data, WSD-trunk recipe, 10B tokens,
#     4 jupiter nodes x 8 H100, allocated) with ONE change: each training document's expert pool is a
#     uniformly RANDOM set of d experts instead of the d experts with the largest summed router score
#     over the document. The pool size d is drawn as in the ladder's EMO (uniform in [16, 512]), so
#     only the pool CONTENT changes. Eval pool stays 512 (all experts) and eval-time selection is
#     unchanged. Implemented by OLMOE3_EMO_POOL_SELECT=random in scripts/sparse_experts/olmoe3_275m.py,
#     which patches EmoRouterV2.forward in-process (the pinned external/OLMo-core is unchanged).
#
#     Checkpoints: /weka/oe-training-default/ryanwang/EMO/debug_validation/olmoe3_275m_emo_randsel_10b
#     (permanent every 5000 steps + final); in-loop v3-small ppl validation (11 sets) every 1000
#     steps and at the end; W&B project emo-extension, tags
#     [pretraining, debug_validation, olmoe3_275m, 512e, emo, pool_select_random, pool_min16, jupiter, 10b].
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/debug_validation/model_scripts/olmoe3_275m_emo_randsel_10b.sh [dry_run]
##############################################################
export OLMOE3_EMO=1
export OLMOE3_EMO_POOL_SELECT=random
export OLMOE3_EMO_MIN_POOL="${OLMOE3_EMO_MIN_POOL:-16}"
export OLMOE3_EXPERIMENT=debug_validation
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_randsel_10b}"
export OLMOE3_SAVE_ROOT="${OLMOE3_SAVE_ROOT:-/weka/oe-training-default/ryanwang/EMO/debug_validation}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-10b}"
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_PREEMPTIBLE=0   # allocated
export OLMOE3_PPL_EVAL_INTERVAL="${OLMOE3_PPL_EVAL_INTERVAL:-1000}"   # in-loop v3-small ppl validation (11 sets) every 1000 steps + at the end
export OLMOE3_FOLLOW="${OLMOE3_FOLLOW:-0}"   # submit and return; watch with beaker CLI
source "$(dirname "${BASH_SOURCE[0]}")/../../sparse_experts/model_scripts/olmoe3_275m_10b.sh" "$@"
