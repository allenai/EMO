##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh"
# DESCRIPTION:
#     olmoe3_squares sub-model g (0..3): the EMO 512e model with layers 2-9 cut down to block-group
#     g's experts (layer 1 whole), initialised from the sliced step-19074 checkpoint (model +
#     optimizer moments; fresh trainer), trained on group g's re-packed documents (its share of the
#     next 10B tokens, training order preserved) with the same EMO loss, 64 x 8192 global batch
#     (1 node x 8 GPUs x 2 seq x 4 accumulation) and constant LR 8e-4 (WSD trunk, 1-step warmup).
#     Permanent checkpoints at the same fractions of progress as the baseline's 20000 / 25000 /
#     30000 / 35000 / 38148 (fractions 926 / 5926 / 10926 / 15926 / 19074 of 19074 steps).
#
#   SQUARE_GROUP=g bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh [dry_run]
##############################################################
: "${SQUARE_GROUP:?set SQUARE_GROUP=0..3}"
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts   # worker-side paths (weka)
SQ="$W/${SQUARES_NAME:-olmoe3_squares}"
LOCAL_SQ="$(git rev-parse --show-toplevel)/sparse_experts/${SQUARES_NAME:-olmoe3_squares}"   # same storage, as seen from this session
tokens="${OLMOE3_TOKENS:-$(python -c "import json; print(json.load(open('$LOCAL_SQ/pack/stats.json'))['tokens_per_group'][$SQUARE_GROUP])")}"
# steps of this run = tokens / 524288; checkpoint at the baseline's fractions of progress
steps=$(( (tokens + 524287) / 524288 ))
fixed=$(python -c "s=$steps; print(','.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f',{s}')")
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO="${OLMOE3_EMO:-1}"
export OLMOE3_TOKENS="$tokens"
export OLMOE3_NUM_NODES=1
export OLMOE3_GROUPS="$SQ/groups.json"
export OLMOE3_GROUP="$SQUARE_GROUP"
export OLMOE3_DATA_PATHS="${OLMOE3_DATA_PATHS:-$SQ/pack/group$SQUARE_GROUP/*.npy}"
export OLMOE3_INIT_FROM="${OLMOE3_INIT_FROM:-$SQ/init/group$SQUARE_GROUP}"
export OLMOE3_FIXED_STEPS="$fixed"
export OLMOE3_WARMUP=1
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_emo_square$SQUARE_GROUP}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-olmoe3_squares,square}"
echo "square $SQUARE_GROUP: $tokens tokens = $steps steps; fixed checkpoints at $fixed"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
