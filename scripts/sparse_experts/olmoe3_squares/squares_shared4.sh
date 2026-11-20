#!/usr/bin/env bash
# Four shared experts (user request 2026-11-19): the EMO 512e recipe with 4 shared experts per MoE block instead of 1
# (olmoe3_275m_emo_sh4_10b.sh: OLMOE3_NUM_SHARED=4, 304.9M active parameters instead of 276.7M), trained from scratch to 10B, then the same
# random-split controls as EMO 512e at sub-model LR 4e-4: K = 4 or 8 random sub-models (random expert groups of 512/K in layers 2-9, layer 1
# and every shared expert whole; seed 2), the 512e controls' random document packs (olmoe3_squares_stdrand / _stdrand8, pack for 10B -> 20B,
# pack2 for 20B -> 30B), merged with token-share weights at the matched points 30,000 (15.7B), 38,148 (20B) and 57,221 (30B); the joint
# baseline continues 10B -> 30B. Merges, every sub-model alone, the start model and the baseline points are scored on the 300B held-out
# sample and the v3-small ppl sets. Comparison arms: olmoe3_squares_emorand_lr4e-4 / olmoe3_squares_emorand8_lr4e-4 (1 shared expert).
#   base : 10B from scratch (4 nodes, keeper) -> start-model evaluations -> joint baseline 10B -> 30B (4 nodes, keeper) -> its three points
#   k4|k8: waits for the 10B checkpoint; slices; packed 8-GPU workers (square_worker.sh, both windows); merges; one 8-GPU evaluation job
#   bash scripts/sparse_experts/olmoe3_squares/squares_shared4.sh base|k4|k8      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
MODE="${1:?base|k4|k8}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; WR=/weka/oe-training-default/ryanwang/EMO; SAMPLE=sample_8k_300b.npz; LR=4e-4
FULL=olmoe3_275m_emo_sh4_10b; BASE=olmoe3_275m_emo_sh4_30b; HRB=runs_heldout300b_emosh4; KEEP=$S/olmoe3_routing/baseline_keeper
SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP $KEEP
case $MODE in
  base) SQN=olmoe3_squares_emosh4rand8; HR=$HRB;;
  k4)   K=4; NWORK="${NWORK:-4}"; PACKSRC=olmoe3_squares_stdrand;  SQN=olmoe3_squares_emosh4rand4; RP=olmoe3_275m_emosh4rand4_square; HR=runs_heldout300b_emosh4rand4;;
  k8)   K=8; NWORK="${NWORK:-4}"; PACKSRC=olmoe3_squares_stdrand8; SQN=olmoe3_squares_emosh4rand8; RP=olmoe3_275m_emosh4rand8_square; HR=runs_heldout300b_emosh4rand8;;
  *) echo "base|k4|k8"; exit 1;;
esac
SQ=$S/$SQN; H=$S/olmoe3_routing/$HR; LOG=$SQ/logs; mkdir -p $LOG $H
say() { echo "$(date -u +%m-%d\ %H:%M) [sh4 $MODE] $*"; }
ended() { beaker experiment get "$1" --format json 2>/dev/null | grep -q '"exited"\|"canceled"'; }
have() { [ -s "$H/$1/none/meta.json" ] || [ -f "$H/$1/none/rank0/DONE" ]; }
hline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none/rank0 --restrict none --batch-size 8 --log-every 100"; }
pline() { echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $1 --out-dir ${2:-$W/$SQN/ppl_validation} --batch-size 8"; }
# run the commands of $CF (built by the caller after reading the job state) as one 8-GPU job reading its own snapshot; 0 = nothing left to run
evaljob() { local key=$1 e tries u; [ -s $CF ] || return 0
  e=$(tail -n 1 $LOG/sh4evals_$key.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1); if [ -n "$e" ] && ! ended $e; then return 1; fi
  if [ -n "$e" ]; then sleep 30; $2; [ -s $CF ] || return 0; fi   # the job just ended: rebuild before resubmitting
  tries=$(cat $LOG/sh4evals_$key.launched 2>/dev/null | wc -l); [ $tries -ge 4 ] && return 1; cp $CF $LOG/sh4evals_${key}_try$tries.txt
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "sh4-$MODE-evals-$key-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/sh4evals_${key}_try$tries.txt 2>&1 | tail -n 1)
  say "evals $key ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/sh4evals_$key.launched; return 1; }
fold() { local d; for d in $H/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done; }
ce() { python -c "import json, os; d = '$H/$1/none/'; print(round(json.load(open(d + ('meta.json' if os.path.exists(d + 'meta.json') else 'rank0/meta.json')))['mean_ce'], 4))" 2>/dev/null; }
keeper() { pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh" > /dev/null || { ( setsid nohup bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh >> $KEEP/log.txt 2>&1 < /dev/null & ); say "baseline keeper started"; }; }
launch_train() { local urlf=$1 script=$2 u=""; [ -f $urlf ] && return 0
  for attempt in 1 2 3 4 5 6; do OLMOE3_FOLLOW=0 bash $script launch > $SP/launch_$(basename $urlf).log 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' $SP/launch_$(basename $urlf).log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $script): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $urlf; }

# =========================== base: 10B from scratch, then the joint baseline to 30B ===========================
if [ $MODE = base ]; then
  [ -f $S/$FULL/step19074/train/rank0.pt ] || launch_train $KEEP/emo_sh4_10b.url scripts/sparse_experts/model_scripts/olmoe3_275m_emo_sh4_10b.sh; keeper
  until [ -f $S/$FULL/step19074/train/rank0.pt ]; do l=$(ls $S/$FULL 2>/dev/null | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n | tail -1); [ "${l:-}" != "${last:-}" ] && { say "10B run: latest checkpoint step ${l:-none}"; last=${l:-}; }; sleep 600; done
  say "10B checkpoint present"
  [ -f $S/$BASE/step57221/train/rank0.pt ] || launch_train $KEEP/emo_sh4_30b.url scripts/sparse_experts/model_scripts/olmoe3_275m_emo_sh4_30b_baseline.sh; keeper
  PS=$WR/claude_outputs/debug_validation/ppl_validation
  build() { local st; : > $CF; have baseline_step19074 || hline baseline_step19074 $W/$FULL/step19074 >> $CF; [ -f claude_outputs/debug_validation/ppl_validation/$FULL/step19074.json ] || pline $W/$FULL/step19074 $PS >> $CF
    for st in 30000 38148 57221; do [ -f $S/$BASE/step$st/train/rank0.pt ] || continue; have baseline_step$st || hline baseline_step$st $W/$BASE/step$st >> $CF; [ -f $SQ/ppl_validation/$BASE/step$st.json ] || pline $W/$BASE/step$st >> $CF; done; }
  # two rounds: the start model as soon as it exists (with whatever baseline points are there), then the rest once the baseline is finished
  for round in start final; do CF=$LOG/sh4evals_base_${round}_cur.txt
    [ $round = final ] && until [ -f $S/$BASE/step57221/train/rank0.pt ]; do l=$(ls $S/$BASE 2>/dev/null | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n | tail -1); [ "${l:-}" != "${last:-}" ] && { say "baseline: latest checkpoint step ${l:-none}"; last=${l:-}; }; sleep 600; done
    while true; do build; evaljob base_$round build && break; [ "$(cat $LOG/sh4evals_base_$round.launched 2>/dev/null | wc -l)" -ge 4 ] && ended "$(tail -n 1 $LOG/sh4evals_base_$round.launched | grep -o '01[A-Z0-9]\{20,\}')" && { say "ABORT: $round evaluations still missing after 4 jobs"; exit 1; }; sleep 300; done
    fold; say "$round evaluations in: start $(ce baseline_step19074), baseline 15.7B $(ce baseline_step30000) 20B $(ce baseline_step38148) 30B $(ce baseline_step57221)"; done
  say done; exit 0
fi

# =========================== k4 / k8: the random-split controls ===========================
until [ -f $S/$FULL/step19074/train/rank0.pt ]; do sleep 600; done
GS=$(seq 0 $((K-1)))
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k $K --seed 2 --num-experts 512
for d in pack pack2; do [ -e $SQ/$d ] || ln -s ../$PACKSRC/$d $SQ/$d; done
for half in $(seq 0 $((K/4-1))); do pids=""; for g in $(seq $((half*4)) $((half*4+3))); do [ -f $SQ/init/group$g/.metadata ] && continue
    { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; } & pids="$pids $!"; done; [ -z "$pids" ] || wait $pids; done
for g in $GS; do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice group$g missing (see $LOG/slice_group$g.log)"; exit 1; }; done; say "start slices present"
declare -a FIN MID FIN2
for g in $GS; do FIN[$g]=$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g] // 524288)"); MID[$g]=$(python -c "print(max(1, round(${FIN[$g]}*10926/19074)))")
  FIN2[$g]=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g] // 524288)"); done
alltrained() { local g; for g in $GS; do [ -f $S/${RP}${g}_w2/step${FIN2[$g]}/train/rank0.pt ] || return 1; done; }
workers() { local i e n u
  for i in $(seq 1 $NWORK); do e=$(tail -n 1 $LOG/worker$i.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1)
    if [ -n "$e" ]; then ended $e || continue; fi
    n=$(cat $LOG/worker$i.launched 2>/dev/null | wc -l); [ $n -ge 8 ] && continue
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "sh4-$MODE-worker$i-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP SQUARES_K=$K WORKER_OFFSET=$(( (i - 1) * K / NWORK )) OLMOE3_LR=$LR OLMOE3_EMO=1 OLMOE3_NUM_SHARED=4 OLMOE3_WANDB_TAGS=$SQN,square,shared4 bash scripts/sparse_experts/olmoe3_squares/square_worker.sh" 2>&1 | tail -n 1)
    say "worker $i (launch $((n+1))): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/worker$i.launched; done; }
until alltrained; do workers; n=$(for g in $GS; do [ -f $S/${RP}$g/step${FIN[$g]}/train/rank0.pt ] && echo w1; [ -f $S/${RP}${g}_w2/step${FIN2[$g]}/train/rank0.pt ] && echo w2; done | wc -l)
  [ "$n" != "${lastn:-}" ] && { say "$n of $((2*K)) square stages finished"; lastn=$n; }; sleep 600; done; say "all $K squares trained to 30B"
merge() { local s=$1 pk=$2; shift 2; [ -f $SQ/merged/match$s/merge_info.json ] && return 0
  local sh; sh=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/$pk/stats.json'))['token_share']))")
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "$*")" --weights "$sh" --out $SQ/merged/match$s --overwrite > $LOG/merge_$s.log 2>&1
  [ -f $SQ/merged/match$s/merge_info.json ] || { say "ABORT: merge at $s failed (see $LOG/merge_$s.log)"; exit 1; }; }
sub_of() { case $1 in 30000) echo "${RP}$2/step${MID[$2]}";; 38148) echo "${RP}$2/step${FIN[$2]}";; 57221) echo "${RP}$2_w2/step${FIN2[$2]}";; esac; }
for s in 30000 38148 57221; do pk=pack; [ $s = 57221 ] && pk=pack2; merge $s $pk $(for g in $GS; do echo "$S/$(sub_of $s $g)"; done); done; say "merged at 15.7B, 20B and 30B"
build() { local s g r; : > $CF
  for s in 30000 38148 57221; do have merged_match$s || hline merged_match$s $W/$SQN/merged/match$s >> $CF; [ -f $SQ/ppl_validation/merged/match$s.json ] || pline $W/$SQN/merged/match$s >> $CF
    for g in $GS; do r=$(sub_of $s $g); have sub${g}_match$s || hline sub${g}_match$s $W/$r >> $CF; [ -f $SQ/ppl_validation/$r.json ] || pline $W/$r >> $CF; done; done; }
CF=$LOG/sh4evals_squares_cur.txt
while true; do build; evaljob squares build && break; [ "$(cat $LOG/sh4evals_squares.launched 2>/dev/null | wc -l)" -ge 4 ] && ended "$(tail -n 1 $LOG/sh4evals_squares.launched | grep -o '01[A-Z0-9]\{20,\}')" && { say "ABORT: evaluations still missing after 4 jobs"; exit 1; }; sleep 300; done
fold; say "merged held-out CE: 15.7B $(ce merged_match30000), 20B $(ce merged_match38148), 30B $(ce merged_match57221)"; say done
