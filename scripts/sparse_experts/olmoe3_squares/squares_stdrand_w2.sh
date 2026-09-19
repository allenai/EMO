#!/usr/bin/env bash
# Window 2 (20B -> 30B tokens) of the random-partition control (report Q3 block B; user request 2026-09-18): the four random
# squares continue from their window-1 finals on a uniformly random quarter each of the NEXT 10B tokens (stream steps 38,147-57,221,
# the same documents block B's window 2 used), constant LR (WSD trunk); merged with equal weights at the five window-2 matched
# points; merged model and every square evaluated on the 300B-token held-out sample.
#   bash scripts/sparse_experts/olmoe3_squares/squares_stdrand_w2.sh      (idempotent; detach it; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_stdrand; SQ=$S/$SQN; SRC=$S/olmoe3_squares_std
FULL=olmoe3_275m_10b; RP=olmoe3_275m_stdrand_square; HR=runs_heldout300b_stdrand; SAMPLE=sample_8k_300b.npz; ST=$S/olmoe3_routing/stream_20b_30b
W2_STEPS=(39073 44073 49073 54073 57221); LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $LOG $SP
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; have $S/olmoe3_routing/$HR/$tag/none || launch "$SQN-w2-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-w2-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
# ---- 1. random document groups of window 2 (same documents as block B's window 2), packs ----
[ -f $SQ/assign2/records_7.npz ] || python scripts/sparse_experts/olmoe3_squares/assign_random.py --src $SRC/assign2 --out $SQ/assign2 --k 4 --seed 1
[ -f $SQ/pack2/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign2 --stream $ST --out $SQ/pack2 --groups $SQ/groups.json 2>&1 | tail -2; say "pack2 done"; }
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack2/stats.json'))['token_share']))")
# ---- 2. init2 = window-1 finals rewritten to the clean layout; window-2 squares ----
declare -a W1F W2_SQ
for g in 0 1 2 3; do W1F[$g]=$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g] // 524288)")
  [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}$g/step${W1F[$g]} --out $SQ/init2/group$g --overwrite 2>&1 | tail -1; say "init2 group$g done"; }
  tokens=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g])")
  W2_SQ[$g]=$(python -c "s=$tokens//524288; print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')")
  if [ ! -f $LOG/square${g}_w2_launched ]; then u=""
    for attempt in 1 2 3 4 5 6; do SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_TOKENS=$tokens OLMOE3_DATA_PATHS="$W/$SQN/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$W/$SQN/init2/group$g/model_and_optim" OLMOE3_RUNNAME=${RP}${g}_w2 OLMOE3_WANDB_TAGS=$SQN,square,w2 OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_std_square.sh launch > $LOG/launch_square${g}_w2.log 2>&1
      u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/launch_square${g}_w2.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
    say "square${g}_w2: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/square${g}_w2_launched; fi
done
say "window-2 square steps: g0 [${W2_SQ[0]}] g1 [${W2_SQ[1]}] g2 [${W2_SQ[2]}] g3 [${W2_SQ[3]}]; shares $SHARES"
# ---- 3. matched points: merge + merged evals; square passes ----
for i in 0 1 2 3 4; do s=${W2_STEPS[$i]}; subs=(); for g in 0 1 2 3; do st=(${W2_SQ[$g]}); subs+=("$S/${RP}${g}_w2/step${st[$i]}"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; done; done; say "window-2 point $s: square checkpoints present"
  [ -f $SQ/merged/match$s/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SHARES" --out $SQ/merged/match$s --overwrite 2>&1 | tail -1 | cut -c1-120
  heldout merged_match$s $W/$SQN/merged/match$s; ppl $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in 0 1 2 3; do st=(${W2_SQ[$g]}); heldout sub${g}_match$s $W/${RP}${g}_w2/step${st[$i]}; done
done
# ---- 4. collect ----
expected=""; for s in "${W2_STEPS[@]}"; do expected="$expected merged_match$s"; for g in 0 1 2 3; do expected="$expected sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$HR/$tag/none; do sleep 120; done; done
for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
python - <<'PY'
import json; from pathlib import Path
H=Path("sparse_experts/olmoe3_routing/runs_heldout300b_stdrand"); B=Path("sparse_experts/olmoe3_routing/runs_heldout300b_std")
ce=lambda d: round(json.load(open(d/"none/meta.json"))["mean_ce"],4) if (d/"none/meta.json").exists() else None
for s in (39073,44073,49073,54073,57221):
    print(s, "baseline", ce(B/f"baseline_step{s}"), "merged(random)", ce(H/f"merged_match{s}"), "squares", [ce(H/f"sub{g}_match{s}") for g in range(4)], "merged(routing)", ce(B/f"merged_match{s}"))
PY
say "random-partition control window 2 done"
