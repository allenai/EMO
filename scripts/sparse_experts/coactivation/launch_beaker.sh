# DESCRIPTION:
#     Expert co-activation extraction for sparse_8of512_10b/step2384 on the held-out 40k-doc set
#     (scripts/sparse_experts/coactivation/PROPOSAL.md). One allocated node, 8 GPUs, torchrun;
#     rank r takes doc_id % 8 == r; rank 0 merges. Two passes in the same job (model loaded once):
#     eval pool = config (512, full routing) and eval pool pinned to 64.
#
#     Pilot:   MAX_DOCS=400 MODE=beaker BEAKER_NO_FOLLOW=1 bash scripts/sparse_experts/coactivation/launch_beaker.sh
#     Full:    MODE=beaker BEAKER_NO_FOLLOW=1 bash scripts/sparse_experts/coactivation/launch_beaker.sh
#     1024:    MODEL=sparse_8of1024_10b MODE=beaker BEAKER_NO_FOLLOW=1 bash scripts/sparse_experts/coactivation/launch_beaker.sh
#     (commit AND push first -- gantry clones from origin)
#
#     Paths are absolute weka paths: on Beaker workers the repo is the gantry clone, NOT the weka
#     tree that holds the checkpoint / docs / outputs.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
# MODEL / STEP select the checkpoint; e.g. MODEL=sparse_8of1024_10b for the 1024-expert arm.
MODEL="${MODEL:-sparse_8of512_10b}"
STEP="${STEP:-2384}"
CHECKPOINT="${CHECKPOINT:-${WEKA_ROOT}/sparse_experts/${MODEL}/step${STEP}}"
DOCS="${DOCS:-${WEKA_ROOT}/sparse_experts/coactivation/docs_40k/docs.jsonl.gz}"
OUT_DIR="${OUT_DIR:-${WEKA_ROOT}/sparse_experts/coactivation/${MODEL}_step${STEP}}"
EVAL_POOLS="${EVAL_POOLS:-config,64}"
MAX_DOCS="${MAX_DOCS:-}"          # set for a pilot (doc_id < MAX_DOCS); empty = all 40k
BATCH_SIZE="${BATCH_SIZE:-4}"

BEAKER_NODES=1
BEAKER_GPUS=8
PREEMPTIBLE=0
BEAKER_TORCHRUN=1

runname="coact_${MODEL#sparse_}_step${STEP}${MAX_DOCS:+_pilot${MAX_DOCS}}"
[ -n "${MAX_DOCS}" ] && OUT_DIR="${OUT_DIR}_pilot${MAX_DOCS}"

launch scripts/sparse_experts/coactivation/extract_coactivation.py "$runname" \
    --checkpoint="${CHECKPOINT}" \
    --docs="${DOCS}" \
    --out-dir="${OUT_DIR}" \
    --eval-pools="${EVAL_POOLS}" \
    --batch-size="${BATCH_SIZE}" \
    ${MAX_DOCS:+--max-docs=${MAX_DOCS}} \
    "$@"
