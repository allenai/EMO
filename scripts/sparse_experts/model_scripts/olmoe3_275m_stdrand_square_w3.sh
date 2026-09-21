##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh"
# DESCRIPTION:
#     Window 3 (30B -> 130B tokens) square g of the random-partition control: the sliced standard-routing model
#     (random expert group g of olmoe3_squares_stdrand, 128 experts per layer 2-9, layer 1 whole) continues from its
#     window-2 final on a CONTIGUOUS quarter of the next 100B tokens of the training stream (stream steps
#     57,221 + g x 47,684 -> + 47,684), read straight from the standard data mix through a finetune-start checkpoint
#     (make_finetune_start.py pins the trainer/data position). The stream is a global shuffle of packed instances, so a
#     contiguous quarter is a uniformly random quarter of the window's documents, like windows 1-2's random split, without
#     extracting and re-packing 100B tokens. Constant LR; checkpoints at the five matched fractions of the window.
#
#   SQUARE_GROUP=g FT_START=<hybrid dir> FT_START_STEP=<57221+g*47684> bash olmoe3_275m_stdrand_square_w3.sh [dry_run]
##############################################################
: "${SQUARE_GROUP:?set SQUARE_GROUP=0..3}"
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
SQN="${SQUARES_NAME:-olmoe3_squares_stdrand}"   # olmoe3_squares_emorand (EMO control) / olmoe3_squares_stdremerge (re-partitioned at 61B)
export OLMOE3_EMO="${OLMOE3_EMO:-0}"
export OLMOE3_GROUPS="$W/$SQN/groups.json"
export OLMOE3_GROUP="$SQUARE_GROUP"
export FT_STEPS="${FT_STEPS:-47684}"
export OLMOE3_FIXED_STEPS="${OLMOE3_FIXED_STEPS:-$(python -c "s=$FT_STEPS; b=${FT_START_STEP:?set FT_START_STEP}; print(','.join(str(b + max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f',{b + s}')")}"
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-${RUN_PREFIX:-olmoe3_275m_stdrand_square}${SQUARE_GROUP}_w3}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-$SQN,square,w3}"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_ft.sh" "$@"
