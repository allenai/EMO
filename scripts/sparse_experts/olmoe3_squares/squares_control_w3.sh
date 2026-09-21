#!/usr/bin/env bash
# Window 3 (30B -> 130B tokens) of the random-partition controls (std: block B, 2026-09-19; emo: block A, 2026-09-20): the four random
# squares continue from their window-2 finals, each on a contiguous quarter of the next 100B tokens of the standard stream
# (see model_scripts/olmoe3_275m_stdrand_square_w3.sh), and the standard baseline continues 30B -> 130B (4 nodes). Merged with
# equal weights at the five matched points; merged model, baseline and every square evaluated on the 300B-token held-out sample
# (steps 572,205+, still far past the 130B end).   bash scripts/sparse_experts/olmoe3_squares/squares_stdrand_w3.sh   (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
V="${1:-std}"; S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz
case $V in
  std) SQN=olmoe3_squares_stdrand; FULL=olmoe3_275m_10b;     RP=olmoe3_275m_stdrand_square; HR=runs_heldout300b_stdrand; HRB=runs_heldout300b_std; BRUN=olmoe3_275m_130b;     B30=olmoe3_275m_30b_1node;     BSCRIPT=olmoe3_275m_130b_baseline.sh;     EMO=0; PPLB=olmoe3_squares_std;;
  emo) SQN=olmoe3_squares_emorand; FULL=olmoe3_275m_emo_10b; RP=olmoe3_275m_emorand_square; HR=runs_heldout300b_emorand; HRB=runs_heldout300b_emo; BRUN=olmoe3_275m_emo_130b; B30=olmoe3_275m_emo_30b_1node; BSCRIPT=olmoe3_275m_emo_130b_baseline.sh; EMO=1; PPLB=olmoe3_squares_emorand;;
  *) echo "std|emo"; exit 1;;
esac
SQ=$S/$SQN; B0=57221; SQ_STEPS=47684; W3_STEPS=(66481 116479 166478 216477 247956)
LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $LOG $SP $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in 1 2 3 4 5 6; do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local hr=$1 tag=$2 ckpt=$3; have $S/olmoe3_routing/$hr/$tag/none || launch "$SQN-w3-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local sqn=$1 ckpt=$2 json=$3 name=$4; [ -f $S/$sqn/ppl_validation/$json ] || launch "$SQN-w3-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$sqn/ppl_validation" --batch-size 8; }
# ---- 1. baseline 30B -> 130B (4 nodes) ----
launch_train $LOG/baseline130_launched $LOG/launch_baseline130.log scripts/sparse_experts/model_scripts/$BSCRIPT
# ---- 2. squares: window-2 finals rewritten -> finetune-start at each square's stream position -> train ----
declare -a W2F ST
for g in 0 1 2 3; do W2F[$g]=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g] // 524288)")
  [ -f $SQ/init3/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}${g}_w2/step${W2F[$g]} --out $SQ/init3/group$g --overwrite 2>&1 | tail -1; say "init3 group$g done"; }
  start=$((B0 + g * SQ_STEPS))
  [ -f $SQ/ft_start/w3_group$g/step$start/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/init3/group$g --train-from $S/$B30/step57221 --start-step $start --out $SQ/ft_start/w3_group$g 2>&1 | tail -1
  ST[$g]=$(python -c "s=$SQ_STEPS; b=$start; print(' '.join(str(b + max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {b + s}')")
  launch_train $LOG/square${g}_w3_launched $LOG/launch_square${g}_w3.log scripts/sparse_experts/model_scripts/olmoe3_275m_stdrand_square_w3.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=$EMO FT_START=$W/$SQN/ft_start/w3_group$g FT_START_STEP=$start FT_STEPS=$SQ_STEPS
done
say "window-3 square steps: g0 [${ST[0]}] g1 [${ST[1]}] g2 [${ST[2]}] g3 [${ST[3]}]"
# ---- 3. matched points: merge + evals (merged, baseline, squares) ----
for i in 0 1 2 3 4; do s=${W3_STEPS[$i]}; subs=(); for g in 0 1 2 3; do st=(${ST[$g]}); subs+=("$S/${RP}${g}_w3/step${st[$i]}"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 600; done; done; say "window-3 point $s: square checkpoints present"
  [ -f $SQ/merged/match$s/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights 0.25,0.25,0.25,0.25 --out $SQ/merged/match$s --overwrite 2>&1 | tail -1 | cut -c1-120
  heldout $HR merged_match$s $W/$SQN/merged/match$s; ppl $SQN $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in 0 1 2 3; do st=(${ST[$g]}); heldout $HR sub${g}_match$s $W/${RP}${g}_w3/step${st[$i]}; done
  until [ -f $S/$BRUN/step$s/train/rank0.pt ]; do sleep 600; done; heldout $HRB baseline_step$s $W/$BRUN/step$s; ppl $PPLB $W/$BRUN/step$s $BRUN/step$s.json baseline-$s
done
# ---- 4. collect ----
expected=""; for s in "${W3_STEPS[@]}"; do expected="$expected $HR/merged_match$s $HRB/baseline_step$s"; for g in 0 1 2 3; do expected="$expected $HR/sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$tag/none; do sleep 300; done; done
for d in $S/olmoe3_routing/$HR/*/none $S/olmoe3_routing/$HRB/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
HR=$HR HRB=$HRB python - <<'PY'
import json; from pathlib import Path
import os; H=Path("sparse_experts/olmoe3_routing")/os.environ["HR"]; B=Path("sparse_experts/olmoe3_routing")/os.environ["HRB"]
ce=lambda d: round(json.load(open(d/"none/meta.json"))["mean_ce"],4) if (d/"none/meta.json").exists() else None
for s in (66481,116479,166478,216477,247956):
    print(s, "baseline", ce(B/f"baseline_step{s}"), "merged(random)", ce(H/f"merged_match{s}"), "squares", [ce(H/f"sub{g}_match{s}") for g in range(4)])
PY
say "random-partition control ($V) window 3 done"
