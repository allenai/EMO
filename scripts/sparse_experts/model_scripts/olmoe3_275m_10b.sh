# DESCRIPTION:
#     The OLMo-core team's OLMoE3 scaling-ladder 275M rung (allenai/scaling-ladders, branch
#     codex/olmoe3-integration-run), trained for a fixed 10B tokens. Runs on the pinned OLMo-core
#     submodule (external/OLMo-core @ 0e5c2d44) with the team's B300 image, NOT this repo's
#     olmo_core; model + recipe live in scripts/sparse_experts/olmoe3_275m.py.
#
#     Model (standard softmax top-16 router, no EMO): d_model 640, 10 layers, 8 heads / 8 KV
#     heads (head_dim 128), 512 routed experts + 1 shared, expert hidden 544, LatentMoE L=2
#     (routed dim 320), dense layer 0 (8*d_model FFN), KDA at 8 layers + NoPE gated full
#     attention at layers 4 and 9. Active 276.7M / active non-embed 212.4M / total 2.608B.
#
#     Recipe (ladder PT defaults): Dolma 3.5 14T from s3://ai2-llm, seq 8192, global batch 64 seq
#     = 524,288 tokens (-> 19,073 steps), AdamW (0.9, 0.95) wd 0.1, clip 1.0, 10% linear warmup
#     then cosine to 10%. LR 1.2e-3: the ladder's WSD sweep found 1.6e-3 best at 8.5B tokens and
#     8e-4 at 17B; 10B sits between. Topology follows the other scripts here: ai2/jupiter,
#     4 nodes x 8 H100 (allocated), EP1, rank micro-batch 2 seq (32 ranks x 2 = 64, no grad
#     accumulation). The ladder itself ran this rung on 4 B300s with FA4 + the CuTe KDA kernel;
#     on H100 the script selects flash-attn 2 and the FLA Triton KDA kernel instead (same math).
#     Checkpoints every 1000 steps (~0.5B tokens) under ${OLMOE3_SAVE_ROOT}/olmoe3_275m_10b.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_10b.sh            # launch (blocks, streams logs)
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_10b.sh dry_run    # print the config only
#
#   B300 (the ladder's own setting): OLMOE3_CLUSTER=ai2/holmes OLMOE3_NUM_NODES=1 OLMOE3_NUM_GPUS=4 \
#     OLMOE3_RANK_MB=16 OLMOE3_ATTN_BACKEND=flash_4 OLMOE3_USE_CUTE_KDA=1 (note: ai2/flex2 has no
#     Holmes allocation, so that needs an unallocated/filler workspace setting).
##############################################################
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"   # worker PYTHONPATH is repo-relative; gantry reads cwd's git state
git submodule update --init external/OLMo-core
unset GH_TOKEN   # the session's GH_TOKEN is invalid; gantry's ref check should fall back to the gh CLI

export OLMOE3_TOKENS="${OLMOE3_TOKENS:-10000000000}"
export OLMOE3_LR="${OLMOE3_LR:-1.2e-3}"
export OLMOE3_NUM_NODES="${OLMOE3_NUM_NODES:-4}"
export OLMOE3_NUM_GPUS="${OLMOE3_NUM_GPUS:-8}"
export OLMOE3_RANK_MB="${OLMOE3_RANK_MB:-2}"
export OLMOE3_ATTN_BACKEND="${OLMOE3_ATTN_BACKEND:-flash_2}"
export OLMOE3_USE_CUTE_KDA="${OLMOE3_USE_CUTE_KDA:-0}"
export OLMOE3_EP_SIZE="${OLMOE3_EP_SIZE:-1}"
export OLMOE3_PREEMPTIBLE=0   # allocated slot: multi-node preemptible gangs queue for days / die mid-run
export OLMOE3_DATA_ROOT=s3://ai2-llm
export OLMOE3_SAVE_ROOT="${OLMOE3_SAVE_ROOT:-/weka/oe-training-default/ryanwang/EMO/sparse_experts}"
export OLMOE3_WORK_DIR=/weka/oe-training-default/ryanwang/dataset-cache
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-10b}"

runname="${OLMOE3_RUNNAME:-olmoe3_275m_10b}"
subcmd="${1:-launch}"; [[ $# -gt 0 ]] && shift
PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python scripts/sparse_experts/olmoe3_275m.py "${subcmd}" "${runname}" "${OLMOE3_CLUSTER:-ai2/jupiter}" "$@"
