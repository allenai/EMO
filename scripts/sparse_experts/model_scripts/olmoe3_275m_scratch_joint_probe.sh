#!/usr/bin/env bash
##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_10b.sh"
# DESCRIPTION:
#     From-scratch probe, joint reference (user request 2026-11-13): the standard-MoE 512e recipe of olmoe3_275m_10b (random init, LR 8e-4,
#     2,000-step warmup, the same stream) run for its first 800 steps only on ONE node (8 GPUs x 2 seq x 4 accumulation = the same 64 x 8192
#     batch), with the random-init model saved as step0 and checkpoints at 10 20 30 50 70 100 150 200 300 400 600 800. step0 is the model
#     the from-scratch squares are sliced from (squares_scratch_probe.sh); the later steps are the joint line of the probe (800 joint steps
#     = the tokens 8 squares consume in 100 steps each). Checkpoints are deleted after evaluation except step0.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_scratch_joint_probe.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=0
export OLMOE3_TOKENS=10000000000
export OLMOE3_NUM_NODES=1
export OLMOE3_PRE_TRAIN_CKPT=1
export OLMOE3_HARD_STOP=800
export OLMOE3_FIXED_STEPS=10,20,30,50,70,100,150,200,300,400,600,800
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_scratch_joint_steps800}"
export OLMOE3_WANDB_TAGS=olmoe3_squares_scratch,first_steps,joint
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}"
