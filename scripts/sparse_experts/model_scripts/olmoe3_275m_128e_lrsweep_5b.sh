#!/usr/bin/env bash
# Pretraining-LR sweep arm for the standard-routing 128e model (user request 2026-09-27): from scratch, 10,000 steps = 5.24B tokens
# (2,000-step warmup then constant peak LR, the WSD trunk as in olmoe3_275m_128e_10b), one jupiter node x 8 H100 (same global batch),
# a single permanent checkpoint at step 10,000 (the 8e-4 arm is olmoe3_275m_128e_10b/step10000). Selection on the separate validation
# sample; the chosen LR is then pretrained from scratch to 130B (olmoe3_275m_128e_lrsel_130b.sh).
#   OLMOE3_LR=2e-4 bash scripts/sparse_experts/model_scripts/olmoe3_275m_128e_lrsweep_5b.sh [dry_run]
: "${OLMOE3_LR:?set OLMOE3_LR}"
export OLMOE3_NUM_EXPERTS=128
export OLMOE3_EMO=0
export OLMOE3_TOKENS=5242880000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_FIXED_STEPS=10000
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_128e_lr${OLMOE3_LR}_5b}"
export OLMOE3_WANDB_TAGS=5b,olmoe3_squares_s128lr,lr_sweep
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}"
