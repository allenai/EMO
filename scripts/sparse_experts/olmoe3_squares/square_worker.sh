#!/usr/bin/env bash
# Cooperative square worker (2026-11-06): one Beaker job (one node, 8 GPUs) that trains the squares of ONE arm back to back, window 1 and
# window 2, instead of one job per square and window. Why: when jupiter is saturated a whole-node grant can take hours, and an arm of 8
# squares x 2 windows would wait for 16 of them; a worker pays the queue once and keeps its node until the arm is finished. Several identical
# workers may be queued for the same arm (allocated and unallocated): each claims a (square, window) stage with an atomic mkdir lock under
# <arm>/logs/locks, heartbeats it every minute, and skips stages that are finished or held by a live worker; a lock whose heartbeat is older
# than 15 minutes (worker preempted or dead) is taken over and the trainer resumes from the run folder's latest checkpoint. A worker that
# finds nothing left exits 0, so surplus workers cost nothing once the arm is done.
#   stage w1 of square g: init/group<g> on pack/group<g>, checkpoints at the mid-window matched step and the final (as squares_lr_sweep.sh)
#   stage w2 of square g: init2/group<g> = rewrite_checkpoint.py of the square's own window-1 final, on pack2/group<g>, final checkpoint only
# Launched as a plain 8-GPU gantry job (olmoe3_beaker_cmd[_cli].py --gpus 8); each stage is its own torchrun. Extra OLMOE3_* variables in
# the job's environment (e.g. OLMOE3_SQUARE_TOP_K) reach the trainer unchanged. Merging and evaluation stay with the driver
# (squares_lr_sweep.sh ... with OLMOE3_W2_EXTERNAL=1).
#   SQUARES_NAME=<arm dir> RUN_PREFIX=<run prefix incl. _square> OLMOE3_LR=4e-4 [SQUARES_K=8] [WORKER_OFFSET=0] bash scripts/sparse_experts/olmoe3_squares/square_worker.sh
set -u
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
SQ="$W/${SQUARES_NAME:?}"; RP="${RUN_PREFIX:?}"; CLUSTER="${CLUSTER:-ai2/jupiter}"; K="${SQUARES_K:-8}"; OFF="${WORKER_OFFSET:-0}"
ME="${BEAKER_JOB_ID:-$(hostname)-$$}"; LOCKS="$SQ/logs/locks"; mkdir -p "$LOCKS"; STALE=900
export OLMOE3_LR="${OLMOE3_LR:?}" OLMOE3_NUM_EXPERTS="${OLMOE3_NUM_EXPERTS:-512}" OLMOE3_EMO="${OLMOE3_EMO:-1}" OLMOE3_NUM_NODES=1 OLMOE3_NUM_GPUS=8 OLMOE3_RANK_MB=2 OLMOE3_EP_SIZE=1
export OLMOE3_ATTN_BACKEND=flash_3 OLMOE3_USE_CUTE_KDA=0 OLMOE3_PREEMPTIBLE=0 OLMOE3_DATA_ROOT=s3://ai2-llm OLMOE3_SAVE_ROOT="$W" OLMOE3_WORK_DIR=/weka/oe-training-default/ryanwang/dataset-cache
export OLMOE3_GROUPS="$SQ/groups.json" OLMOE3_WARMUP=1
# environment the gantry-launched training jobs carry (see `beaker experiment spec` of any square job)
export NUM_NODES=1 OLMO_SHARED_FS=1 LOG_FILTER_TYPE=local_rank0_only OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OLMO_SYMM_VDEV2D_AUTO_BUILD=0 TORCH_CUDA_ARCH_LIST=9.0 TORCH_LOGS=recompiles,graph_breaks PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels S3_PROFILE=S3 R2_PROFILE=R2 WEKA_PROFILE=WEKA
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p ~/.aws && printf "[S3]\naws_access_key_id=%s\naws_secret_access_key=%s\n" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
TAGS="${OLMOE3_WANDB_TAGS:-$SQUARES_NAME,square,lr_sweep}"
say() { echo "== $(date -u +%m-%d\ %H:%M) [worker $ME] $*"; }
tok() { python -c "import json; print(json.load(open('$SQ/$1/stats.json'))['tokens_per_group'][$2])"; }
have() { [ -f "$1/train/rank0.pt" ]; }
claim() { local l="$LOCKS/$1" age
  if mkdir "$l" 2>/dev/null; then echo "$ME" > "$l/owner"; touch "$l/beat"; return 0; fi
  age=$(( $(date +%s) - $(stat -c %Y "$l/beat" 2>/dev/null || stat -c %Y "$l" 2>/dev/null || date +%s) ))
  if [ "$age" -gt $STALE ]; then say "lock $1 is stale (${age}s, owner $(cat "$l/owner" 2>/dev/null)): taking over"; rm -rf "$l"; mkdir "$l" 2>/dev/null && { echo "$ME" > "$l/owner"; touch "$l/beat"; return 0; }; fi
  return 1; }
HB=""
beat_start() { local l="$LOCKS/$1"; ( while [ -d "$l" ]; do touch "$l/beat" 2>/dev/null; sleep 60; done ) & HB=$!; }
release() { [ -n "$HB" ] && kill "$HB" 2>/dev/null; HB=""; [ "$(cat "$LOCKS/$1/owner" 2>/dev/null)" = "$ME" ] && rm -rf "$LOCKS/$1"; }
train() { local run=$1; say "training $run: $OLMOE3_TOKENS tokens, checkpoints at $OLMOE3_FIXED_STEPS"
  torchrun --standalone --nproc-per-node 8 scripts/sparse_experts/olmoe3_275m.py train "$run" "$CLUSTER"; }
all_done() { local g; for g in $(seq 0 $((K-1))); do have "$W/${RP}${g}_w2/step$(( $(tok pack2 $g) / 524288 ))" || return 1; done; }
fails=0
while ! all_done; do progressed=0
  for i in $(seq 0 $((K-1))); do g=$(( (i + OFF) % K ))
    t1=$(tok pack $g); fin=$(( t1 / 524288 )); mid=$(python -c "print(max(1, round($fin*10926/19074)))"); t2=$(tok pack2 $g); fin2=$(( t2 / 524288 ))
    if ! have "$W/${RP}$g/step$fin" && claim g${g}_w1; then beat_start g${g}_w1; progressed=1
      export OLMOE3_GROUP=$g OLMOE3_TOKENS=$t1 OLMOE3_FIXED_STEPS="$mid,$fin" OLMOE3_DATA_PATHS="$SQ/pack/group$g/*.npy" OLMOE3_INIT_FROM="$SQ/init/group$g" OLMOE3_WANDB_TAGS="$TAGS"
      train "${RP}$g" || { say "${RP}$g FAILED (exit $?)"; fails=$((fails+1)); }
      release g${g}_w1
    fi
    if have "$W/${RP}$g/step$fin" && ! have "$W/${RP}${g}_w2/step$fin2" && claim g${g}_w2; then beat_start g${g}_w2; progressed=1
      if [ ! -f "$SQ/init2/group$g/model_and_optim/.metadata" ]; then say "init2 group$g from ${RP}$g/step$fin"
        python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src "$W/${RP}$g/step$fin" --out "$SQ/init2/group$g" --overwrite 2>&1 | tail -n 2; fi
      if [ -f "$SQ/init2/group$g/model_and_optim/.metadata" ]; then
        export OLMOE3_GROUP=$g OLMOE3_TOKENS=$t2 OLMOE3_FIXED_STEPS="$fin2" OLMOE3_DATA_PATHS="$SQ/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$SQ/init2/group$g/model_and_optim" OLMOE3_WANDB_TAGS="$TAGS,w2"
        train "${RP}${g}_w2" || { say "${RP}${g}_w2 FAILED (exit $?)"; fails=$((fails+1)); }
      else say "init2 group$g could not be written"; fails=$((fails+1)); fi
      release g${g}_w2
    fi
    [ $fails -ge 3 ] && { say "3 failed stages: giving up"; exit 1; }
  done
  [ $progressed = 0 ] && { say "nothing to claim (other workers hold the remaining stages); waiting"; sleep 120; }
done
say "arm $SQUARES_NAME complete"; exit 0
