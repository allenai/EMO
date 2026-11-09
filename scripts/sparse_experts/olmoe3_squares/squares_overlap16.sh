#!/usr/bin/env bash
# Overlapping sub-models (user request 2026-11-09): EMO 512e from the 10B checkpoint, sub-model LR 4e-4, 16 random groups of 64 experts per
# layer 2-9 in which EVERY EXPERT BELONGS TO TWO GROUPS (partition_overlap.py: two independent random 8-way partitions stacked; layer 1 whole),
# i.e. 16 sub-models of the 8-square size whose expert sets overlap. The window-1 and window-2 documents (10B -> 20B -> 30B) are split
# uniformly at random 16 ways (assign_random.py on the random controls' assignment records), so every expert still sees 1/8 of the documents
# through its two copies and the arm costs the same GPU time as the 8-square arm. Merge: experts side by side, each expert the token-share-
# weighted mean of its two owners' copies (merge_models.py), shared parameters averaged over the 16 with token shares. Matched points 30000
# (15.7B, mid-window) / 38148 (20B) / 57221 (30B): merged model on the reporting held-out sample + ppl sets, every sub-model alone (CE only).
# Training = 8 cooperative 8-GPU workers (square_worker.sh, SQUARES_K=16), evaluations = one 8-GPU batch job per point (eval_batch_8gpu.sh).
#   bash scripts/sparse_experts/olmoe3_squares/squares_overlap16.sh      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
K=16; COPIES=2; SEED=3; LR=4e-4; WORKERS="${WORKERS:-8}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_10b; SAMPLE=sample_8k_300b.npz
SQN=olmoe3_squares_emorand16ov; RP=olmoe3_275m_emorand16ov_square; HR=runs_heldout300b_emorand16ov; SRC=olmoe3_squares_stdrand8   # assignment records of the window-1 / window-2 documents
SQ=$S/$SQN; LOG=$SQ/logs; mkdir -p $SQ $LOG $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [overlap16] $*"; }
GS=$(seq 0 $((K-1)))
# 1. groups, document packs, start slices
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_overlap.py --out $SQ/groups.json --k $K --copies $COPIES --seed $SEED
[ -f $SQ/assign/records_7.npz ]  || python scripts/sparse_experts/olmoe3_squares/assign_random.py --src $S/$SRC/assign  --out $SQ/assign  --k $K --seed $SEED
[ -f $SQ/assign2/records_7.npz ] || python scripts/sparse_experts/olmoe3_squares/assign_random.py --src $S/$SRC/assign2 --out $SQ/assign2 --k $K --seed $((SEED + 10))
[ -f $SQ/pack/stats.json ]  || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign  --stream $S/olmoe3_routing/stream_10b_20b --out $SQ/pack  --groups $SQ/groups.json > $LOG/pack.log 2>&1; say "pack: $(tail -n 1 $LOG/pack.log | cut -c1-160)"; } &
[ -f $SQ/pack2/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign2 --stream $S/olmoe3_routing/stream_20b_30b --out $SQ/pack2 --groups $SQ/groups.json > $LOG/pack2.log 2>&1; say "pack2: $(tail -n 1 $LOG/pack2.log | cut -c1-160)"; } &
n=0; for g in $GS; do [ -f $SQ/init/group$g/.metadata ] && continue
  { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=4 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; say "slice group$g: $(tail -n 1 $LOG/slice_group$g.log | cut -c1-100)"; } &
  n=$((n+1)); [ $((n % 8)) = 0 ] && wait; done; wait
for g in $GS; do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: slice group$g missing"; exit 1; }; done; [ -f $SQ/pack/stats.json ] && [ -f $SQ/pack2/stats.json ] || { say "ABORT: packs missing"; exit 1; }
shares_of() { python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$1'))['token_share']))"; }
tok() { python -c "import json; print(json.load(open('$SQ/$1/stats.json'))['tokens_per_group'][$2])"; }
merge_k() { local out=$1 weights=$2; shift 2; [ -f $out/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "$*")" --weights "$weights" --out $out --overwrite 2>&1 | tail -n 1 | cut -c1-160; }
# 2. merge check: the untrained slices merged back must reproduce the start model (every expert's two copies are identical)
if [ ! -f $LOG/init_merge_check.json ]; then subs=(); for g in $GS; do subs+=("$SQ/init/group$g"); done
  merge_k $SQ/merged/init_check "$(shares_of $SQ/pack/stats.json)" "${subs[@]}"
  PYTHONPATH=external/OLMo-core/src python - <<PY > $LOG/init_merge_check.json 2>&1
import json, torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
a="$S/$FULL/step19074/model_and_optim"; b="$SQ/merged/init_check/model_and_optim"; mx=0.0; n=0
for k in get_checkpoint_metadata(b).state_dict_metadata:
    if not k.endswith(".main"): continue
    x=next(load_keys(a,[k])); y=next(load_keys(b,[k])); mx=max(mx, float((x.float()-y.float()).abs().max())); n+=1
print(json.dumps(dict(n_tensors=n, max_abs_diff=mx, ok=mx < 1e-5)))
PY
  say "init merge check: $(cat $LOG/init_merge_check.json | tail -n 1)"; rm -rf $SQ/merged/init_check; fi
grep -q '"ok": true' $LOG/init_merge_check.json || { say "ABORT: init merge check failed"; exit 1; }
# 3. training: cooperative workers (each claims (square, window) stages; a worker that finds nothing left exits)
for i in $(seq 0 $((WORKERS-1))); do mark=$LOG/worker_$i.launched; [ -f $mark ] && continue
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "emorand16ov-w$i" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_LR=$LR SQUARES_K=$K WORKER_OFFSET=$((i * 2)) OLMOE3_WANDB_TAGS=$SQN,square,overlap16 bash scripts/sparse_experts/olmoe3_squares/square_worker.sh" 2>&1 | tail -n 1)
  say "worker $i: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark || sleep 60; done
# 4. per matched point: wait for the 16 checkpoints, merge, one 8-GPU evaluation job (merged full pass + ppl sets, every sub-model CE only)
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
fold() { [ -f "$1/meta.json" ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_routing/merge_routing.py "$1" > /dev/null 2>&1; }
point() { local s=$1 pk=$2 which=$3 suffix=$4; local subs=() g st t fin
  for g in $GS; do t=$(tok $pk $g); fin=$((t / 524288)); [ $which = mid ] && st=$(python -c "print(max(1, round($fin*10926/19074)))") || st=$fin
    until [ -f $S/${RP}$g$suffix/step$st/train/rank0.pt ]; do sleep 300; done; subs+=("$S/${RP}$g$suffix/step$st"); done; say "point $s: 16 checkpoints present"
  merge_k $SQ/merged/match$s "$(shares_of $SQ/$pk/stats.json)" "${subs[@]}"
  local CF=$LOG/evals_match$s.txt; : > $CF
  have $S/olmoe3_routing/$HR/merged_match$s/none || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$SQN/merged/match$s --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/merged_match$s/none/rank0 --restrict none --batch-size 8 --log-every 100" >> $CF
  [ -f $SQ/ppl_validation/merged/match$s.json ] || echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $W/$SQN/merged/match$s --out-dir $W/$SQN/ppl_validation --batch-size 8" >> $CF
  for g in $GS; do have $S/olmoe3_routing/$HR/sub${g}_match$s/none || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint ${subs[$g]/#$S/$W} --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/sub${g}_match$s/none --restrict none --batch-size 8 --log-every 100 --ce-only" >> $CF; done
  if [ -s $CF ] && [ ! -f $LOG/evals_match$s.launched ]; then u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "emorand16ov-evals-$s" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_match$s.txt 2>&1 | tail -n 1)
    say "evals $s ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/evals_match$s.launched; fi; }
point 30000 pack mid ""; point 38148 pack fin ""; point 57221 pack2 fin _w2
for s in 30000 38148 57221; do for p in merged_match$s $(for g in $GS; do echo sub${g}_match$s; done); do d=$S/olmoe3_routing/$HR/$p/none; until have $d; do sleep 300; done; fold $d; done
  until [ -f $SQ/ppl_validation/merged/match$s.json ]; do sleep 300; done
  say "point $s: merged CE $(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/merged_match$s/none/meta.json'))['mean_ce'],4))"), sub-models mean $(python -c "import json; print(round(sum(json.load(open(f'$S/olmoe3_routing/$HR/sub{g}_match$s/none/meta.json'))['mean_ce'] for g in range($K))/$K,4))")"; done
say done
