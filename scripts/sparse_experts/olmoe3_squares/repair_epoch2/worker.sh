#!/usr/bin/env bash
# Runs INSIDE one Beaker job (1 node x 8 GPUs, launched by launch.sh): resume each square of a variant from its latest
# checkpoint before the tainted final and train to the correct final step s = floor(tokens / 524288), in place
# (save folder = the run dir; the trainer resumes from the latest checkpoint it finds there, so launch.sh must have
# renamed the old final to step<s+1>_epoch2 first). Same env as train_multi.sh.
#   bash worker.sh <variant> <g:tokens:s:s_old:resume> ...
set -u
V=$1; shift
source "$(dirname "${BASH_SOURCE[0]}")/variants.sh"; variant_env "$V" || exit 1
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQ="$W/$SQN"; CLUSTER="${CLUSTER:-ai2/jupiter}"
export OLMOE3_NUM_EXPERTS=512 OLMOE3_EMO="$EMO" OLMOE3_LR=8e-4 OLMOE3_SCHEDULER=wsd OLMOE3_NUM_NODES=1 OLMOE3_NUM_GPUS=8 OLMOE3_RANK_MB=2 OLMOE3_EP_SIZE=1
export OLMOE3_ATTN_BACKEND=flash_3 OLMOE3_USE_CUTE_KDA=0 OLMOE3_DATA_ROOT=s3://ai2-llm OLMOE3_SAVE_ROOT="$W" OLMOE3_WORK_DIR=/weka/oe-training-default/ryanwang/dataset-cache
export OLMOE3_GROUPS="$SQ/groups.json" OLMOE3_WARMUP=1 OLMOE3_WANDB_TAGS="$SQN,square,epoch2_repair"
[ -n "$EXTRA_ENV" ] && export $EXTRA_ENV
unset OLMOE3_INIT_FROM   # resume (model + optimizer + trainer/data-loader state) from the run dir, not a fresh init
export NUM_NODES=1 OLMO_SHARED_FS=1 LOG_FILTER_TYPE=local_rank0_only OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OLMO_SYMM_VDEV2D_AUTO_BUILD=0 TORCH_CUDA_ARCH_LIST=9.0 TORCH_LOGS=recompiles,graph_breaks PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels S3_PROFILE=S3 R2_PROFILE=R2 WEKA_PROFILE=WEKA
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p ~/.aws && printf "[S3]\naws_access_key_id=%s\naws_secret_access_key=%s\n" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
status=0
for spec in "$@"; do
  IFS=: read -r g tokens s old resume <<< "$spec"; run="${RP}$g"
  if [ -f "$W/$run/step$s/train/rank0.pt" ]; then echo "== $run: step$s already present, skipping"; continue; fi
  [ -d "$W/$run/step$old" ] && { echo "== $run: tainted final step$old still present (launch.sh renames it) -- refusing"; status=1; continue; }
  [ -f "$W/$run/step$resume/train/rank0.pt" ] || { echo "== $run: resume checkpoint step$resume missing"; status=1; continue; }
  export OLMOE3_TOKENS="$tokens" OLMOE3_GROUP="$g" OLMOE3_DATA_PATHS="$SQ/$PACK/group$g/*.npy" OLMOE3_FIXED_STEPS="$(fixed_steps $s)"
  echo "== $(date -u +%H:%M) $run: resume from step$resume -> step$s ($tokens tokens; fixed $OLMOE3_FIXED_STEPS)"
  torchrun --standalone --nproc-per-node 8 scripts/sparse_experts/olmoe3_275m.py train "$run" "$CLUSTER" || { echo "== $run FAILED (exit $?)"; status=1; continue; }
  [ -f "$W/$run/step$s/train/rank0.pt" ] || { echo "== $run: step$s NOT written"; status=1; }
done
exit $status
