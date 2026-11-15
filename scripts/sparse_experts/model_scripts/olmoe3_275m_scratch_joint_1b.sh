#!/usr/bin/env bash
##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_scratch_joint_probe.sh"
# DESCRIPTION:
#     Joint trunk of the branch-point experiment (user request 2026-11-15): the standard-MoE 512e recipe of olmoe3_275m_10b from the random
#     init (LR 8e-4, 2,000-step warmup, the same stream) for the first 1B tokens (1,907 steps) plus the 100M tokens (191 steps) a branch
#     trains for, on ONE node. Checkpoints at the nine branch points 10 25 50 100 200 400 800 1300 1907 (log-spaced; kept) and 48 / 96 / 144 /
#     191 steps after each (the joint reference of every branch; deleted after scoring). squares_branch.sh splits every branch point into 4 and 8
#     sub-models for the next 100M tokens.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_scratch_joint_1b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=0
export OLMOE3_TOKENS=10000000000
export OLMOE3_NUM_NODES=1
export OLMOE3_PRE_TRAIN_CKPT=1
export OLMOE3_HARD_STOP=2098
export OLMOE3_FIXED_STEPS=10,25,50,58,73,98,100,106,121,146,148,154,169,194,196,200,201,216,241,244,248,291,296,344,391,400,448,496,544,591,800,848,896,944,991,1300,1348,1396,1444,1491,1907,1955,2003,2051,2098
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_scratch_joint_1b}"
export OLMOE3_WANDB_TAGS=olmoe3_squares_branch,joint
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}"
