#!/usr/bin/env bash
# Window 2 (20B -> 30B tokens) of the standard-MoE squares (report Q3 block B), user request 2026-09-16: keep training the
# four squares separately on their documents of the NEXT 10B tokens (stream steps 38147-57221) from their window-1 final
# checkpoints, merge at the same five progress fractions, and compare with the baseline continued to 30B. All held-out
# numbers (both windows, re-evaluated) use a fresh sample from 300B tokens into the stream (steps 572205-572605).
#   bash scripts/sparse_experts/olmoe3_squares/squares_window2_std.sh        (idempotent; detach it)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_std; SQ=$S/$SQN; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
ST=$S/olmoe3_routing/stream_20b_30b; SAMPLE=olmoe3_routing/sample_8k_300b.npz; HR=runs_heldout300b_std; START=std_step19074
FULL=olmoe3_275m_10b; BRUN=olmoe3_275m_30b_1node; B1RUN=olmoe3_275m_20b_1node; PW=claude_outputs/olmoe3_routing/${SQN}_w2
LOG=$SQ/logs; mkdir -p $LOG $S/olmoe3_routing/$HR $PW
W1_STEPS=(20000 25000 30000 35000 38148); W1_FINAL=(3698 5170 5656 4554)
W1_SQ=("180 1149 2118 3088 3698" "251 1606 2961 4317 5170" "275 1757 3240 4723 5656" "221 1415 2609 3802 4554")
W2_STEPS=(39073 44073 49073 54073 57221)
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; local R=$S/olmoe3_routing/$HR/$tag/none; have $R || launch "$SQN-w2-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-w2-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
mergeall() { for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done; }
# ---- 1. held-out sample from 300B tokens ----
until [ -f $S/olmoe3_routing/stream_300b_slice/manifest.json ]; do sleep 120; done
[ -f $S/$SAMPLE ] || { python scripts/sparse_experts/olmoe3_routing/sample_instances.py --stream $S/olmoe3_routing/stream_300b_slice --out $S/$SAMPLE 2>&1 | tail -1; say "sample_8k_300b.npz written"; }
# ---- 2. re-evaluate window 1 on the new sample (start, baseline points, merged points, square passes) ----
heldout $START $W/$FULL/step19074
for i in 0 1 2 3 4; do s=${W1_STEPS[$i]}; bs=$s; [ $s = 38148 ] && bs=38147
  heldout baseline_step$s $W/$B1RUN/step$bs; heldout merged_match$s $W/$SQN/merged/match$s
  for g in 0 1 2 3; do st=(${W1_SQ[$g]}); heldout sub${g}_match$s $W/olmoe3_275m_square$g/step${st[$i]}; done
done
# ---- 3. window-2 assignment + pack ----
until [ -f $ST/manifest.json ]; do sleep 300; done; say "stream 20b-30b extracted"
for r in 0 1 2 3 4 5 6 7; do [ -f $SQ/assign2/records_$r.npz ] || [ -f $LOG/assign2_r$r.launched ] || { launch "$SQN-w2-assign-r$r" python scripts/sparse_experts/olmoe3_squares/assign_docs.py --checkpoint "$W/$FULL/step19074" --stream "$W/olmoe3_routing/stream_20b_30b" --groups "$W/$SQN/groups.json" --out "$W/$SQN/assign2" --rank $r --world 8 --batch-size 8; touch $LOG/assign2_r$r.launched; }; done
for r in 0 1 2 3 4 5 6 7; do until [ -f $SQ/assign2/records_$r.npz ]; do sleep 120; done; done; say "window-2 assignment complete"
[ -f $SQ/pack2/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign2 --stream $ST --out $SQ/pack2 --groups $SQ/groups.json 2>&1 | tail -2; say "pack2 done"; }
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack2/stats.json'))['token_share']))")
# ---- 4. init2 = window-1 finals rewritten to the clean layout; train squares window 2 ----
declare -a W2_SQ
for g in 0 1 2 3; do
  [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/olmoe3_275m_square$g/step${W1_FINAL[$g]} --out $SQ/init2/group$g --overwrite 2>&1 | tail -1; say "init2 group$g done"; }
  tokens=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g])")
  W2_SQ[$g]=$(python -c "import math; s=math.ceil($tokens/524288); print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')")
  if [ ! -f $LOG/square${g}_w2_launched ]; then u=""
    for attempt in 1 2 3 4 5 6; do SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_TOKENS=$tokens OLMOE3_DATA_PATHS="$W/$SQN/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$W/$SQN/init2/group$g" OLMOE3_RUNNAME=olmoe3_275m_square${g}_w2 OLMOE3_WANDB_TAGS=$SQN,square,w2 OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_std_square.sh launch > $LOG/launch_square${g}_w2.log 2>&1
      u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/launch_square${g}_w2.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
    say "square${g}_w2: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/square${g}_w2_launched; fi
done
say "window-2 square steps: g0 [${W2_SQ[0]}] g1 [${W2_SQ[1]}] g2 [${W2_SQ[2]}] g3 [${W2_SQ[3]}]; shares $SHARES"
# ---- 5. window-2 matched points: merge + evaluate; baseline evaluate; square passes ----
for i in 0 1 2 3 4; do s=${W2_STEPS[$i]}; subs=(); for g in 0 1 2 3; do st=(${W2_SQ[$g]}); subs+=("$S/olmoe3_275m_square${g}_w2/step${st[$i]}"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; done; done; say "window-2 point $s: square checkpoints present"
  [ -f $SQ/merged/match$s/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SHARES" --out $SQ/merged/match$s --overwrite 2>&1 | tail -1 | cut -c1-120
  heldout merged_match$s $W/$SQN/merged/match$s; ppl $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in 0 1 2 3; do st=(${W2_SQ[$g]}); heldout sub${g}_match$s $W/olmoe3_275m_square${g}_w2/step${st[$i]}; done
  until [ -f $S/$BRUN/step$s/train/rank0.pt ]; do sleep 300; done; heldout baseline_step$s $W/$BRUN/step$s; ppl $W/$BRUN/step$s $BRUN/step$s.json baseline-$s
done
# ---- 6. collect + piecewise for both windows on the new sample ----
for tag in $S/olmoe3_routing/$HR/*/none; do until [ -f $tag/rank0/DONE ] || [ -f $tag/meta.json ]; do sleep 120; done; done; mergeall
for s in "${W1_STEPS[@]}" "${W2_STEPS[@]}"; do HELDOUT_DIR=$HR SQUARES_NAME=$SQN START_TAG=$START python scripts/sparse_experts/olmoe3_squares/piecewise_eval.py --name match$s --out-dir $PW | head -2; done
say "window 2 done"
