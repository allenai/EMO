#!/usr/bin/env bash
# From-scratch squares over the first 10B tokens (user request 2026-11-13; the 100-step probe squares_scratch_probe.sh showed the squares
# and their merge train): the standard-MoE 512e random init (step0 of olmoe3_275m_scratch_joint_steps800) cut into K random squares
# (olmoe3_squares_scratch<K>: groups, packs of the run's own 0-10B documents, start slices), each square trained on its pack with peak
# LR <lr>, load-balancing weight <lb> and a linear warmup of 2000/K steps, never merged during training. At the joint baseline's
# (olmoe3_275m_10b) checkpoints 5000 / 10000 / 15000 / 19074 (2.6B / 5.2B / 7.9B / 10B tokens) the squares' token-matched checkpoints are
# merged (token-share weights) and the merge and every square alone are scored on the reporting held-out sample (CE only).
# A <stop> argument (a matched step, e.g. 5000) ends every square there: the LR / LB ablation arms. `baseline` scores the joint run's four
# checkpoints. Cooperative single-window workers (square_worker.sh ONE_WINDOW=1). After a point is scored its merged weights and non-final
# square checkpoints are deleted; ablation arms keep nothing, the full arm keeps the squares' finals and the final merge.
#   bash scripts/sparse_experts/olmoe3_squares/squares_scratch.sh k4|k8 [lr=8e-4] [lb=1e-2] [stop]   |   ... baseline      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k4|k8|baseline}"; LR="${2:-8e-4}"; LB="${3:-1e-2}"; STOP="${4:-}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; INIT=olmoe3_275m_scratch_joint_steps800/step0; BASE=olmoe3_275m_10b
HR=runs_heldout300b_scratch; H=$S/olmoe3_routing/$HR; mkdir -p $H; ALL="5000 10000 15000 19074"
evline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none --restrict none --batch-size 8 --log-every 100 --ce-only --max-first-ce 100"; }
done_ev() { [ -f $H/$1/none/meta.json ]; }
ce() { python -c "import json; print(round(json.load(open('$H/$1/none/meta.json'))['mean_ce'],4))" 2>/dev/null; }
launch_evals() { local name=$1 cf=$2 mark=$3; [ -s $cf ] || return 0; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name-$(date +%m%d%H%M)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh ${cf/$S/$W} 2>&1 | tail -n 1)
  say "evals $name ($(wc -l < $cf) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
if [ $KV = baseline ]; then LOG=$S/olmoe3_squares_scratch/logs/baseline; mkdir -p $LOG; say() { echo "$(date -u +%m-%d\ %H:%M) [scratch baseline] $*"; }
  CF=$LOG/evals.txt; : > $CF; for m in $ALL; do done_ev baseline_step$m || evline baseline_step$m $W/$BASE/step$m >> $CF; done; launch_evals scratch-baseline $CF $LOG/evals.launched
  for m in $ALL; do until done_ev baseline_step$m; do sleep 120; done; done; say "joint baseline: $(for m in $ALL; do echo -n "$m=$(ce baseline_step$m) "; done)"; exit 0; fi
case $KV in k4) K=4;; k8) K=8;; *) echo "k4|k8|baseline"; exit 1;; esac
ARM=k${K}_lr${LR}_lb${LB}; SQN=olmoe3_squares_scratch$K; SQ=$S/$SQN; RP=olmoe3_275m_scratch${K}_lr${LR}_lb${LB}_square; LOG=$SQ/logs/$ARM; mkdir -p $LOG; WARM=$((2000 / K))
say() { echo "$(date -u +%m-%d\ %H:%M) [scratch $ARM${STOP:+ stop $STOP}] $*"; }
PTS=""; for m in $ALL; do [ -z "$STOP" ] || [ $m -le $STOP ] && PTS="$PTS $m"; done; LAST=${PTS##* }; NW=${NW:-$([ -z "$STOP" ] && echo 4 || echo $((K / 4)))}
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] && [ -f $SQ/pack/stats.json ] || { say "ABORT: $SQ lacks packs or start slices (run squares_scratch_probe.sh first)"; exit 1; }; done
sstep() { python -c "import json; fin=json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$1]//524288; print(fin if $2==19074 else max(1, round(fin*$2/19074)))"; }
# ---- 1. workers ----
for i in $(seq 0 $((NW-1))); do mark=$LOG/worker$i.launched; [ -f $mark ] && continue
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "scratch-$ARM-w$i-$(date +%m%d%H%M)" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP SQUARES_K=$K WORKER_OFFSET=$((i * K / NW)) ONE_WINDOW=1 W1_FRACS=5000,10000,15000 W1_STOP_FRAC=$STOP LOCK_TAG=$ARM OLMOE3_EMO=0 OLMOE3_LR=$LR OLMOE3_LB_WEIGHT=$LB OLMOE3_WARMUP=$WARM OLMOE3_WANDB_TAGS=$SQN,square,scratch,$ARM bash scripts/sparse_experts/olmoe3_squares/square_worker.sh" 2>&1 | tail -n 1)
  say "worker $i: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; done
# ---- 2. matched points: merge, score, prune ----
SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
for m in $PTS; do tag=${ARM}_merged_match$m; MD=$SQ/merged_$ARM/match$m
  need=0; done_ev $tag || need=1; for g in $(seq 0 $((K-1))); do done_ev ${ARM}_sub${g}_match$m || need=1; done
  if [ $need = 1 ]; then
    for g in $(seq 0 $((K-1))); do until [ -f $S/${RP}$g/step$(sstep $g $m)/train/rank0.pt ]; do sleep 300; done; done; say "match $m: all $K squares have their checkpoints"
    if [ ! -f $MD/merge_info.json ]; then subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$(sstep $g $m)"); done
      PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$INIT --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $MD --overwrite 2>&1 | tail -n 1 | cut -c1-120; fi
    [ -f $MD/merge_info.json ] || { say "ABORT: merge at $m failed"; exit 1; }
    CF=$LOG/evals_match$m.txt; : > $CF; done_ev $tag || evline $tag $W/$SQN/merged_$ARM/match$m >> $CF
    for g in $(seq 0 $((K-1))); do done_ev ${ARM}_sub${g}_match$m || evline ${ARM}_sub${g}_match$m $W/${RP}$g/step$(sstep $g $m) >> $CF; done
    launch_evals scratch-$ARM-m$m $CF $LOG/evals_match$m.launched
    until done_ev $tag; do sleep 120; done; for g in $(seq 0 $((K-1))); do until done_ev ${ARM}_sub${g}_match$m; do sleep 120; done; done
  fi
  say "match $m: merged $(ce $tag); squares alone $(for g in $(seq 0 $((K-1))); do echo -n "$(ce ${ARM}_sub${g}_match$m) "; done); joint baseline $(ce baseline_step$m)"
  if [ -n "$STOP" ] || [ $m != $LAST ]; then rm -rf "${MD:?}/model_and_optim"; for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g/step$(sstep $g $m)"; done; fi
done
say done
