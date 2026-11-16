#!/usr/bin/env bash
# K-means expert groups (user request 2026-11-15): the EMO 512e 8-square control at sub-model LR 4e-4 with ONE change -- the experts of
# layers 2-9 are grouped by k-means on their router rows (partition_kmeans.py: 8 groups of unequal size, clusters matched across layers by
# centroid similarity; layer 1 whole) instead of at random. Documents: exactly the random packs of the random-group control
# (olmoe3_squares_emorand8/pack), so the comparison arm is olmoe3_squares_emorand8_lr4e-4. Window 10B -> 20B only: the squares checkpoint at
# the mid-window matched point (baseline step 30000, 15.7B) and at the window final (38148, 20B); both are merged (token-share weights)
# and the merges and every square alone are scored on the 300B held-out sample and the v3-small ppl sets. One 8-GPU job trains the eight
# squares back to back (square_worker.sh), one 8-GPU job runs all evaluations (eval_batch_8gpu.sh).
#   bash scripts/sparse_experts/olmoe3_squares/squares_kmeans.sh      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_10b; SAMPLE=sample_8k_300b.npz; LR=4e-4; K=8
SQN=olmoe3_squares_emokm8; SQ=$S/$SQN; RP=olmoe3_275m_emokm8_square; HR=runs_heldout300b_emokm8; H=$S/olmoe3_routing/$HR; LOG=$SQ/logs; mkdir -p $LOG $H
say() { echo "$(date -u +%m-%d\ %H:%M) [kmeans8] $*"; }
GS=$(seq 0 $((K-1)))
[ -f $SQ/groups.json ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/partition_kmeans.py --checkpoint $S/$FULL/step19074 --out $SQ/groups.json > $LOG/partition.log 2>&1
[ -e $SQ/pack ] || ln -s ../olmoe3_squares_emorand8/pack $SQ/pack
for half in 0 1; do pids=""; for g in $(seq $((half*4)) $((half*4+3))); do [ -f $SQ/init/group$g/.metadata ] && continue
    { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; } & pids="$pids $!"; done; [ -z "$pids" ] || wait $pids; done
for g in $GS; do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice group$g missing"; exit 1; }; done; say "start slices present"
declare -a FIN MID; for g in $GS; do t=$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g])"); FIN[$g]=$((t / 524288)); MID[$g]=$(python -c "print(max(1, round(${FIN[$g]}*10926/19074)))"); done
alltrained() { local g; for g in $GS; do [ -f $S/${RP}$g/step${FIN[$g]}/train/rank0.pt ] || return 1; done; }
if ! alltrained && [ ! -f $LOG/worker.launched ]; then
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "emokm8-squares-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_LR=$LR OLMOE3_EMO=1 ONE_WINDOW=1 W1_FRACS=10926 OLMOE3_WANDB_TAGS=$SQN,square,kmeans bash scripts/sparse_experts/olmoe3_squares/square_worker.sh" 2>&1 | tail -n 1)
  say "squares worker: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/worker.launched; fi
until alltrained; do sleep 300; done; say "all $K squares trained"
SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
for point in 30000:MID 38148:FIN; do s=${point%%:*}; arr=${point##*:}; subs=(); for g in $GS; do eval "st=\${$arr[$g]}"; subs+=("$S/${RP}$g/step$st"); done
  [ -f $SQ/merged/match$s/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged/match$s --overwrite > $LOG/merge_$s.log 2>&1
  [ -f $SQ/merged/match$s/merge_info.json ] || { say "ABORT: merge at $s failed"; exit 1; }; done; say "merged at 30000 and 38148"
# ---- evaluations: held-out pass (full routing statistics, as the control) + ppl sets, for both merges and every square at both points ----
have() { [ -s "$H/$1/none/meta.json" ] || [ -f "$H/$1/none/rank0/DONE" ]; }
hline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none/rank0 --restrict none --batch-size 8 --log-every 100"; }
pline() { echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $1 --out-dir $W/$SQN/ppl_validation --batch-size 8"; }
build() { : > $CF; local s arr g st
  for point in 30000:MID 38148:FIN; do s=${point%%:*}; arr=${point##*:}
    have merged_match$s || hline merged_match$s $W/$SQN/merged/match$s >> $CF; [ -f $SQ/ppl_validation/merged/match$s.json ] || pline $W/$SQN/merged/match$s >> $CF
    for g in $GS; do eval "st=\${$arr[$g]}"; have sub${g}_match$s || hline sub${g}_match$s $W/${RP}$g/step$st >> $CF; [ -f $SQ/ppl_validation/${RP}$g/step$st.json ] || pline $W/${RP}$g/step$st >> $CF; done; done; }
CF=$LOG/evals.txt; tries=0
while true; do build; [ -s $CF ] || break
  e=$(grep -o '01[A-Z0-9]\{20,\}' $LOG/evals.launched 2>/dev/null | head -1)
  if [ -z "$e" ] || beaker experiment get $e --format json 2>/dev/null | grep -q '"exited"'; then
    [ -n "$e" ] && { sleep 60; build; [ -s $CF ] || break; }; tries=$((tries+1)); [ $tries -gt 4 ] && { say "ABORT: evaluations still missing after 3 resubmissions"; exit 1; }
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "emokm8-evals-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals.txt 2>&1 | tail -n 1)
    say "evals ($(wc -l < $CF) commands, try $tries): $u"; echo "$u" > $LOG/evals.launched; fi
  sleep 300; done
for d in $H/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
ce() { python -c "import json; print(round(json.load(open('$H/$1/none/meta.json'))['mean_ce'],4))" 2>/dev/null; }
say "merged held-out CE: 15.7B $(ce merged_match30000), 20B $(ce merged_match38148)"; say done
