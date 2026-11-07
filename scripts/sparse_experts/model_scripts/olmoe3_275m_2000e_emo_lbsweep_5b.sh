##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_10b.sh"
# DESCRIPTION:
#     Load-balancing-weight sweep arm of the EMO 2000e model (user request 2026-11-07, the LR sweep's protocol): the 2000e EMO recipe from
#     scratch to step 10,000 = 5.24B tokens at LR 8e-4 (2,000-step warmup then constant) with lb_loss_weight $OLMOE3_LB_WEIGHT instead of the
#     recipe's 0.01, 4 nodes, EP=2 sync_1d. One checkpoint at step 10,000, evaluated like the LR sweep arms (big_lb_sweep.sh).
#
#   OLMOE3_LB_WEIGHT=<w> bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_lbsweep_5b.sh [dry_run]
##############################################################
: "${OLMOE3_LB_WEIGHT:?set OLMOE3_LB_WEIGHT}"
export OLMOE3_LR=8e-4
export OLMOE3_NUM_EXPERTS=2000
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_EMO=1
export OLMOE3_TOKENS=5242880000
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_FIXED_STEPS=10000
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_2000e_emo_lb${OLMOE3_LB_WEIGHT}_5b}"
export OLMOE3_WANDB_TAGS=5b,olmoe3_squares_emo2000lb,lb_sweep
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}"
