#!/usr/bin/env bash
# Cycle worker of the re-partition-every-~100M-tokens run (squares_repartition100m.sh; user request 2026-11-15). One 8-GPU job that walks
# the cycles of <arm>/plan.tsv (rows: cycle square window start end): for cycle c it (prep) draws fresh random expert groups (seed 1000+c),
# merges the previous cycle's K squares and cuts the merge into the new K squares in memory (remerge_reslice.py, Adam state carried; cycle 1
# cuts the start model), pins each square's trainer/data position (make_finetune_start.py on the control square's trainer state: square g
# always continues on document pack g, steps start..end), then (train) runs each square for its ~24 steps as its own 8-GPU torchrun and
# saves one checkpoint. Cycles listed in <arm>/keep.tsv (cycle, name) also write the merged model to merged/<name>; their squares are never
# deleted here. Everything else of a cycle is deleted two cycles later. Several identical workers may run: stages (prep of a cycle, each
# square, the final merge) are claimed with mkdir locks + heartbeats (stale after 10 min), as in square_worker.sh. W&B is off (1600 runs).
#   SQUARES_NAME=<arm dir> RUN_PREFIX=<prefix> FULL=<start model run> OLMOE3_LR=4e-4 [OLMOE3_EMO_POOL_SELECT=mix:0.2] bash scripts/sparse_experts/olmoe3_squares/repartition_worker.sh
set -u
source "$(dirname "${BASH_SOURCE[0]}")/ensure_env.sh"
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
SQ="$W/${SQUARES_NAME:?}"; RP="${RUN_PREFIX:?}"; FULLD="$W/${FULL:?}/step19074"; CLUSTER="${CLUSTER:-ai2/jupiter}"; K=8; P=scripts/sparse_experts/olmoe3_squares
ME="${BEAKER_JOB_ID:-$(hostname)-$$}"; LOCKS="$SQ/logs/locks"; mkdir -p "$LOCKS" "$SQ/groups" "$SQ/cyc" "$SQ/merged"; STALE=600
export OLMOE3_LR="${OLMOE3_LR:?}" OLMOE3_NUM_EXPERTS=512 OLMOE3_EMO=1 OLMOE3_SCHEDULER=wsd OLMOE3_NUM_NODES=1 OLMOE3_NUM_GPUS=8 OLMOE3_RANK_MB=2 OLMOE3_EP_SIZE=1 OLMOE3_WARMUP=1
export OLMOE3_ATTN_BACKEND=flash_3 OLMOE3_USE_CUTE_KDA=0 OLMOE3_PREEMPTIBLE=0 OLMOE3_DATA_ROOT=s3://ai2-llm OLMOE3_SAVE_ROOT="$W" OLMOE3_WORK_DIR=/weka/oe-training-default/ryanwang/dataset-cache
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-$SQUARES_NAME,square,repartition100m}"
export NUM_NODES=1 OLMO_SHARED_FS=1 LOG_FILTER_TYPE=local_rank0_only OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OLMO_SYMM_VDEV2D_AUTO_BUILD=0 TORCH_CUDA_ARCH_LIST=9.0 TORCH_LOGS=recompiles,graph_breaks PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels S3_PROFILE=S3 R2_PROFILE=R2 WEKA_PROFILE=WEKA
export PYTHONPATH="external/OLMo-core/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p ~/.aws && printf "[S3]\naws_access_key_id=%s\naws_secret_access_key=%s\n" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
say() { echo "== $(date -u +%m-%d\ %H:%M:%S) [rp100m $ME] $*"; }
NC=$(awk '{ if ($1 > m) m = $1 } END { print m }' "$SQ/plan.tsv")
row() { awk -v c="$1" -v g="$2" '$1 == c && $2 == g { print $3, $4, $5 }' "$SQ/plan.tsv"; }   # -> window start end
run_of() { echo "${RP}_c$1_g$2"; }
fin_of() { local r; r=($(row $1 $2)); echo "$W/$(run_of $1 $2)/step${r[2]}"; }
sq_done() { [ -f "$(fin_of $1 $2)/train/rank0.pt" ]; }
all_sq() { local g; for g in $(seq 0 $((K-1))); do sq_done $1 $g || return 1; done; }
keep_name() { awk -v c="$1" '$1 == c { print $2 }' "$SQ/keep.tsv"; }
claim() { local l="$LOCKS/$1" age
  if mkdir "$l" 2>/dev/null; then echo "$ME" > "$l/owner"; touch "$l/beat"; return 0; fi
  age=$(( $(date +%s) - $(stat -c %Y "$l/beat" 2>/dev/null || stat -c %Y "$l" 2>/dev/null || date +%s) ))
  if [ "$age" -gt $STALE ]; then say "lock $1 is stale (${age}s, owner $(cat "$l/owner" 2>/dev/null)): taking over"; rm -rf "${LOCKS:?}/${1:?}"; mkdir "$l" 2>/dev/null && { echo "$ME" > "$l/owner"; touch "$l/beat"; return 0; }; fi
  return 1; }
HB=""
beat_start() { local l="$LOCKS/$1"; ( while [ -d "$l" ]; do touch "$l/beat" 2>/dev/null; sleep 30; done ) & HB=$!; }
release() { [ -n "$HB" ] && kill "$HB" 2>/dev/null; HB=""; [ "$(cat "$LOCKS/$1/owner" 2>/dev/null)" = "$ME" ] && rm -rf "${LOCKS:?}/${1:?}"; }
src_args() { local p=$1 g r subs="" wts=""   # the merge of cycle p (p = 0: the start model)
  if [ $p = 0 ]; then echo "--full $FULLD"; return; fi
  for g in $(seq 0 $((K-1))); do r=($(row $p $g)); subs="$subs,$(fin_of $p $g)"; wts="$wts,$((r[2] - r[1]))"; done
  echo "--old-groups $SQ/groups/c$p.json --subs ${subs#,} --weights ${wts#,}"; }
prep() { local c=$1 p=$(( $1 - 1 )) g r kn extra=""; local D="$SQ/cyc/c$c"
  [ -f "$SQ/groups/c$c.json" ] || python $P/partition_random.py --out "$SQ/groups/c$c.json" --k $K --seed $((1000 + c)) > /dev/null || return 1
  kn=$(keep_name $p); [ -n "$kn" ] && [ ! -f "$SQ/merged/$kn/merge_info.json" ] && extra="--save-merged $SQ/merged/$kn"
  rm -rf "$D"; mkdir -p "$D"
  python $P/remerge_reslice.py --full-config "$FULLD" $(src_args $p) --new-groups "$SQ/groups/c$c.json" --out "$D/init" $extra 2>&1 | tail -n 1 | cut -c1-80 || return 1
  for g in $(seq 0 $((K-1))); do r=($(row $c $g)); [ -f "$D/init/group$g/model_and_optim/.metadata" ] || return 1
    python $P/make_finetune_start.py --model "$D/init/group$g" --train-from "$SQ/templates/w${r[0]}/group$g" --start-step ${r[1]} --out "$D/ft/group$g" > /dev/null || return 1; done
  touch "$D/PREP_DONE"
  local q=$(( c - 2 )); if [ $q -ge 1 ]; then rm -rf "$SQ/cyc/c$q"; [ -z "$(keep_name $q)" ] && for g in $(seq 0 $((K-1))); do rm -rf "$W/$(run_of $q $g)"; done; fi; return 0; }
train() { local c=$1 g=$2 r; r=($(row $c $g)); local pack=pack; [ ${r[0]} = 2 ] && pack=pack2
  export OLMOE3_GROUPS="$SQ/groups/c$c.json" OLMOE3_GROUP=$g OLMOE3_TOKENS=$(( r[2] * 524288 )) OLMOE3_FIXED_STEPS=${r[2]} OLMOE3_DATA_PATHS="$SQ/$pack/group$g/*.npy"
  rm -rf "$W/$(run_of $c $g)"
  torchrun --standalone --nproc-per-node 8 scripts/sparse_experts/olmoe3_275m.py train "$(run_of $c $g)" "$CLUSTER" --trainer.load_path="$SQ/cyc/c$c/ft/group$g/step${r[1]}" --trainer.load_strategy=always --trainer.callbacks.checkpointer.max_checkpoints=null --trainer.callbacks.wandb.enabled=false > "/results/train_c${c}_g$g.log" 2>&1
  local rc=$?; grep -a "train/CE loss\|step=" "/results/train_c${c}_g$g.log" | tail -n 1 | cut -c1-200; return $rc; }
fails=0
while [ ! -f "$SQ/DONE" ]; do did=0
  # shared filesystem full (2026-11-16: 0 bytes free, every write failed and the workers gave up): wait instead of failing
  until [ "$(df --output=avail -BG "$W" | tail -n 1 | tr -dc 0-9)" -gt 300 ]; do say "less than 300 GB free on the shared filesystem: waiting"; sleep 600; done
  cur=0; for c in $(ls "$SQ/cyc" 2>/dev/null | sed 's/^c//' | sort -n); do [ -f "$SQ/cyc/c$c/PREP_DONE" ] && cur=$c; done
  if [ $cur -ge 1 ] && ! all_sq $cur; then
    for g in $(seq 0 $((K-1))); do sq_done $cur $g && continue
      if claim c${cur}_g$g; then beat_start c${cur}_g$g; did=1; sq_done $cur $g || { t0=$(date +%s); train $cur $g && sq_done $cur $g && say "cycle $cur square $g: steps $(row $cur $g | cut -d' ' -f2-) done in $(( $(date +%s) - t0 ))s" || { say "cycle $cur square $g FAILED: $(tail -n 3 /results/train_c${cur}_g$g.log | tr '\n' ' ' | cut -c1-300)"; fails=$((fails+1)); }; }; release c${cur}_g$g; fi
      [ $fails -ge 4 ] && { say "4 failed stages: giving up"; exit 1; }; done
  elif [ $cur -ge $NC ]; then
    if claim final; then beat_start final; did=1; kn=$(keep_name $NC)
      [ -f "$SQ/merged/$kn/merge_info.json" ] || python $P/remerge_reslice.py --full-config "$FULLD" $(src_args $NC) --no-slice --save-merged "$SQ/merged/$kn" 2>&1 | tail -n 1 | cut -c1-80
      [ -f "$SQ/merged/$kn/merge_info.json" ] && { touch "$SQ/DONE"; say "final merge $kn written: run complete"; } || { say "final merge FAILED"; fails=$((fails+1)); }; release final; fi
  else n=$((cur + 1))
    if claim prep_c$n; then beat_start prep_c$n; did=1; t0=$(date +%s)
      if [ -f "$SQ/cyc/c$n/PREP_DONE" ]; then :; elif prep $n; then say "cycle $n prepared in $(( $(date +%s) - t0 ))s"; else say "cycle $n prep FAILED"; fails=$((fails+1)); sleep 30; fi; release prep_c$n; fi
  fi
  [ $fails -ge 4 ] && { say "4 failed stages: giving up"; exit 1; }
  [ $did = 0 ] && sleep 15
done
say "arm $SQUARES_NAME complete"; exit 0
