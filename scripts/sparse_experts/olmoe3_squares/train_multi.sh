#!/usr/bin/env bash
# Train several olmoe3_squares sub-models SEQUENTIALLY inside one Beaker job (one node, 8 GPUs), to pay the
# Beaker queue once instead of once per sub-model. Launched through olmoe3_beaker_cmd.py --gpus 8 (plain
# gantry job, no torchrun wrapper); each sub-model is run with its own torchrun.
#
#   SQUARES_NAME=olmoe3_squares_k8 RUN_PREFIX=olmoe3_275m_emo_k8_square OLMOE3_EMO=1 \
#     bash scripts/sparse_experts/olmoe3_squares/train_multi.sh <group> [<group> ...]
# Per-group tokens come from <SQUARES_NAME>/pack/stats.json; fixed checkpoints at the baseline's progress
# fractions; OLMOE3_TOKENS_OVERRIDE (smoke) forces a token count for every group.
set -u
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
SQ="$W/${SQUARES_NAME:?}"; RP="${RUN_PREFIX:?}"; CLUSTER="${CLUSTER:-ai2/jupiter}"
export OLMOE3_NUM_EXPERTS=512 OLMOE3_EMO="${OLMOE3_EMO:-1}" OLMOE3_LR=8e-4 OLMOE3_SCHEDULER=wsd OLMOE3_NUM_NODES=1 OLMOE3_NUM_GPUS=8 OLMOE3_RANK_MB=2 OLMOE3_EP_SIZE=1
export OLMOE3_ATTN_BACKEND=flash_3 OLMOE3_USE_CUTE_KDA=0 OLMOE3_DATA_ROOT=s3://ai2-llm OLMOE3_SAVE_ROOT="${OLMOE3_SAVE_ROOT:-$W}" OLMOE3_WORK_DIR=/weka/oe-training-default/ryanwang/dataset-cache
export OLMOE3_GROUPS="$SQ/groups.json" OLMOE3_WARMUP=1 OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-$SQUARES_NAME,square}"
# environment the gantry-launched training jobs carry (see `beaker experiment spec` of any square job)
export NUM_NODES=1 OLMO_SHARED_FS=1 LOG_FILTER_TYPE=local_rank0_only OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OLMO_SYMM_VDEV2D_AUTO_BUILD=0 TORCH_CUDA_ARCH_LIST=9.0 TORCH_LOGS=recompiles,graph_breaks PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels S3_PROFILE=S3 R2_PROFILE=R2 WEKA_PROFILE=WEKA
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p ~/.aws && printf "[S3]\naws_access_key_id=%s\naws_secret_access_key=%s\n" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
status=0
for g in "$@"; do
  tokens="${OLMOE3_TOKENS_OVERRIDE:-$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g])")}"
  steps=$(( tokens / 524288 ))   # floor: only full batches exist (ceil ran one epoch-2 step; repaired 2026-09-17)
  fixed=$(python -c "s=$steps; print(','.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f',{s}')")
  export OLMOE3_TOKENS="$tokens" OLMOE3_GROUP="$g" OLMOE3_DATA_PATHS="$SQ/pack/group$g/*.npy" OLMOE3_INIT_FROM="$SQ/init/group$g" OLMOE3_FIXED_STEPS="$fixed"
  run="${RP}$g"
  if [ -f "$OLMOE3_SAVE_ROOT/$run/step$steps/train/rank0.pt" ]; then echo "== $run already complete, skipping"; continue; fi
  echo "== $(date -u +%H:%M) training $run: $tokens tokens = $steps steps; fixed checkpoints $fixed"
  torchrun --standalone --nproc-per-node 8 scripts/sparse_experts/olmoe3_275m.py train "$run" "$CLUSTER" || { echo "== $run FAILED (exit $?)"; status=1; }
done
exit $status
