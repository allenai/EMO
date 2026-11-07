#!/usr/bin/env bash
# First-steps probe of the random squares (user request 2026-11-07): train each square of an arm for only N steps from its start slice, with a
# permanent checkpoint after EVERY step, as an exact prefix of the real square run (same tokens, data order, constant LR; OLMOE3_HARD_STOP stops
# the trainer after N steps). One 8-GPU gantry job trains the listed squares back to back (each its own torchrun); the driver
# (squares_firststeps.sh) merges the K squares after every step and evaluates each merge.
#   SQUARES_NAME=<arm dir> RUN_PREFIX=<run prefix incl. _square> OLMOE3_LR=4e-4 STEPS=10 bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh <group> [<group> ...]
set -u
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
SQ="$W/${SQUARES_NAME:?}"; RP="${RUN_PREFIX:?}"; CLUSTER="${CLUSTER:-ai2/jupiter}"; N="${STEPS:?}"
export OLMOE3_LR="${OLMOE3_LR:?}" OLMOE3_NUM_EXPERTS="${OLMOE3_NUM_EXPERTS:-512}" OLMOE3_EMO="${OLMOE3_EMO:-1}" OLMOE3_NUM_NODES=1 OLMOE3_NUM_GPUS=8 OLMOE3_RANK_MB=2 OLMOE3_EP_SIZE=1
export OLMOE3_ATTN_BACKEND=flash_3 OLMOE3_USE_CUTE_KDA=0 OLMOE3_PREEMPTIBLE=0 OLMOE3_DATA_ROOT=s3://ai2-llm OLMOE3_SAVE_ROOT="$W" OLMOE3_WORK_DIR=/weka/oe-training-default/ryanwang/dataset-cache
export OLMOE3_GROUPS="$SQ/groups.json" OLMOE3_WARMUP=1 OLMOE3_HARD_STOP=$N OLMOE3_FIXED_STEPS="$(seq -s, 1 $N)" OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-$SQUARES_NAME,square,first_steps}"
export NUM_NODES=1 OLMO_SHARED_FS=1 LOG_FILTER_TYPE=local_rank0_only OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OLMO_SYMM_VDEV2D_AUTO_BUILD=0 TORCH_CUDA_ARCH_LIST=9.0 TORCH_LOGS=recompiles,graph_breaks PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels S3_PROFILE=S3 R2_PROFILE=R2 WEKA_PROFILE=WEKA
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p ~/.aws && printf "[S3]\naws_access_key_id=%s\naws_secret_access_key=%s\n" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
say() { echo "== $(date -u +%m-%d\ %H:%M) [steps] $*"; }
status=0
for g in "$@"; do
  run="${RP}$g"; tokens=$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g])")
  if [ -f "$W/$run/step$N/train/rank0.pt" ]; then say "$run has step $N already, skipping"; continue; fi
  export OLMOE3_GROUP=$g OLMOE3_TOKENS=$tokens OLMOE3_DATA_PATHS="$SQ/pack/group$g/*.npy" OLMOE3_INIT_FROM="$SQ/init/group$g"
  say "training $run for $N steps (schedule of the full $tokens-token window), checkpoint after every step"
  torchrun --standalone --nproc-per-node 8 scripts/sparse_experts/olmoe3_275m.py train "$run" "$CLUSTER" || { say "$run FAILED (exit $?)"; status=1; }
done
exit $status
