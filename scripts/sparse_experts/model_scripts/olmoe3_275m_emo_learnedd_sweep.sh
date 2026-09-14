##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_10b.sh"
# DESCRIPTION:
#     2000-step (1.05B-token) hyper-parameter sweep arm for the learned-d EMO router. One config per
#     invocation, named by its knobs; in-loop v3-small ppl eval (predicted-d routing) at steps 1000 and
#     2000; final checkpoint only. Same data order / recipe as the 10B arms, so train CE is comparable
#     step-for-step with olmoe3_275m_emo_10b (uniform pools) and olmoe3_275m_10b (standard).
#
#   OLMOE3_LD_TEMP=2 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=500 \
#     bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_sweep.sh [dry_run]
##############################################################
export OLMOE3_TOKENS=$((2000 * 524288))
export OLMOE3_LD_TEMP="${OLMOE3_LD_TEMP:-2.0}"
export OLMOE3_LD_LAMBDA="${OLMOE3_LD_LAMBDA:-0.01}"
export OLMOE3_LD_WARMUP="${OLMOE3_LD_WARMUP:-500}"
export OLMOE3_LD_LR_MULT="${OLMOE3_LD_LR_MULT:-10}"
export OLMOE3_PPL_EVAL_INTERVAL=1000
export OLMOE3_FIXED_STEPS=2000
tag="T${OLMOE3_LD_TEMP}_l${OLMOE3_LD_LAMBDA}_w${OLMOE3_LD_WARMUP}_m${OLMOE3_LD_LR_MULT}${OLMOE3_LD_INIT:+_i${OLMOE3_LD_INIT}}${OLMOE3_LD_SUFFIX:-}"
# OLMOE3_LD_CONTROL=1: the plain uniform-pool EMO router under the identical 2000-step recipe (reference)
if [[ "${OLMOE3_LD_CONTROL:-0}" == "1" ]]; then export OLMOE3_EMO_LEARNED_D=0; tag="control_uniform"; fi
export OLMOE3_RUNNAME="olmoe3_275m_emo_learnedd_sweep_${tag}"
export OLMOE3_WANDB_TAGS="learnedd_sweep"
export OLMOE3_SAVE_ROOT=/weka/oe-training-default/ryanwang/EMO/sparse_experts/learnedd_sweep
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_emo_learnedd_10b.sh" "$@"
