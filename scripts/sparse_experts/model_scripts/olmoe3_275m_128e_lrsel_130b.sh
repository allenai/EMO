#!/usr/bin/env bash
# The standard-routing 128e model pretrained FROM SCRATCH to 130B at the LR chosen by the 5B sweep (user request 2026-09-27): the
# second 128e baseline line of Q5 box 5C next to the 8e-4 one (olmoe3_275m_128e_10b -> olmoe3_275m_128e_130b), saving at 10B and at
# every matched checkpoint of the controls. 4 jupiter nodes x 8 H100, allocated (baseline_keeper.sh relaunches it after the 8-h cut).
#   OLMOE3_LR=<lr> bash scripts/sparse_experts/model_scripts/olmoe3_275m_128e_lrsel_130b.sh [dry_run]
: "${OLMOE3_LR:?set OLMOE3_LR}"
export OLMOE3_NUM_EXPERTS=128
export OLMOE3_EMO=0
export OLMOE3_TOKENS=130000000000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=10000,19074,20000,25000,30000,35000,38148,39073,44073,49073,54073,57221,66481,116479,166478,216477,247956
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_128e_lr${OLMOE3_LR}_130b}"
export OLMOE3_WANDB_TAGS=130b,olmoe3_squares_s128lr,lr_selected
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}"
