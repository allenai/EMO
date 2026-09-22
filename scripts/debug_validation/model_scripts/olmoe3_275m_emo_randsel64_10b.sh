#!/usr/bin/env bash
# PARENT: scripts/debug_validation/model_scripts/olmoe3_275m_emo_randsel_10b.sh
# DESCRIPTION:
#     Same as olmoe3_275m_emo_randsel_10b (random expert set per training document) but the pool size
#     is drawn uniformly from [64, 512] instead of [16, 512]: 64 = the 8-sub-model size, the smallest
#     pool in case random 16-expert pools are too unstable to train on. Everything else identical.
#     Checkpoints: /weka/oe-training-default/ryanwang/EMO/debug_validation/olmoe3_275m_emo_randsel64_10b
#     W&B tags [..., pool_select_random, pool_min64, ...].
#
#   bash scripts/debug_validation/model_scripts/olmoe3_275m_emo_randsel64_10b.sh [dry_run]
##############################################################
export OLMOE3_EMO_MIN_POOL=64
export OLMOE3_RUNNAME=olmoe3_275m_emo_randsel64_10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_emo_randsel_10b.sh" "$@"
