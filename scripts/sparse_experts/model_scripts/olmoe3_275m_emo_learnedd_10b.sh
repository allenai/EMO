##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh"
# DESCRIPTION:
#     EMO 512-expert arm with a LEARNED per-document expert-pool size d (scripts/sparse_experts/emo_learned_d.py):
#     a d_head per MoE layer predicts each document's pool in [16, 512] from its mean hidden state, the
#     forward uses the hard pool round(d), an STE mask gives the LM loss a boundary-only gradient into d,
#     and a normalised size penalty lambda_d pushes pools small. Optional warm-up (OLMOE3_LD_WARMUP steps)
#     starts unrestricted and phases the restriction in. Everything else = olmoe3_275m_emo_10b (WSD trunk,
#     LR 8e-4, 10B tokens, 5000-step checkpoints); one node.
#     Hyper-parameters (defaults = the 2026-09-14 2k-step sweep winner, learnedd/launch_sweep.sh + sweep_table.py):
#       OLMOE3_LD_SIGNAL=coverage (the STE LM gradient alone collapses d to top_k in every arm, even at lambda_d=0;
#       the router-mass coverage term gives a tunable equilibrium), OLMOE3_LD_LAMBDA_COV=1, OLMOE3_LD_LAMBDA=1.0
#       (mass threshold 1/(1*496) = 0.002 per expert -> pools ~110-130 with 2/3 of documents <= 64 at step 2000,
#       v3-small CE 3.710 vs uniform-pool control 3.705), OLMOE3_LD_TEMP=2, OLMOE3_LD_WARMUP=500 (unrestricted
#       start, restriction phased in), OLMOE3_LD_LR_MULT=10; OLMOE3_LD_INIT (initial d), OLMOE3_LD_EVAL (predicted|fixed).
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_EMO_LEARNED_D="${OLMOE3_EMO_LEARNED_D:-1}"
export OLMOE3_LD_TEMP="${OLMOE3_LD_TEMP:-2.0}"
export OLMOE3_LD_LAMBDA="${OLMOE3_LD_LAMBDA:-1.0}"
export OLMOE3_LD_WARMUP="${OLMOE3_LD_WARMUP:-500}"
export OLMOE3_LD_SIGNAL="${OLMOE3_LD_SIGNAL:-coverage}"
export OLMOE3_LD_LAMBDA_COV="${OLMOE3_LD_LAMBDA_COV:-1.0}"
export OLMOE3_LD_EVAL="${OLMOE3_LD_EVAL:-predicted}"
export OLMOE3_LD_LR_MULT="${OLMOE3_LD_LR_MULT:-10}"
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-1}"
export OLMOE3_PPL_EVAL_INTERVAL="${OLMOE3_PPL_EVAL_INTERVAL:-5000}"
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_learnedd_10b}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-10b}"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
