##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_10b.sh"
# DESCRIPTION:
#     Pretraining-LR sweep arm of the EMO 2000e model (user request 2026-10-20, like olmoe3_275m_128e_lrsweep_5b.sh): the 2000e EMO
#     recipe from scratch to step 10,000 = 5.24B tokens at peak LR $OLMOE3_LR (2,000-step warmup then constant, the WSD trunk), 4 nodes, EP=2 sync_1d.
#     One checkpoint at step 10,000, evaluated on the selection sample, the reporting sample and the ppl sets (big_lr_sweep.sh).
#
#   OLMOE3_LR=<lr> bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_lrsweep_5b.sh [dry_run]
##############################################################
: "${OLMOE3_LR:?set OLMOE3_LR}"
export OLMOE3_NUM_EXPERTS=2000
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_EMO=1
export OLMOE3_TOKENS=5242880000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=10000
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_2000e_emo_lr${OLMOE3_LR}_5b}"
export OLMOE3_WANDB_TAGS=5b,olmoe3_squares_emo2000lr,lr_sweep
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}"
