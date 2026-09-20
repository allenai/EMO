#!/usr/bin/env bash
# Random-partition control for the EMO 512e squares (report Q3 block A; user request 2026-09-20): the SAME random expert groups
# and the SAME random document packs as the standard-MoE control (olmoe3_squares_stdrand: groups.json seed 0, pack / pack2), applied
# to the EMO 10B checkpoint: four EMO squares 10B -> 20B (window 1) then 20B -> 30B (window 2, each continuing from its own
# window-1 final), merged with equal weights at the ten matched points; the EMO baseline continued 20B -> 30B; merged model,
# baseline, start model and every square evaluated on the 300B-token held-out sample.
#   bash scripts/sparse_experts/olmoe3_squares/squares_emorand.sh      (idempotent; detach it; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_emorand; SQ=$S/$SQN; SRC=olmoe3_squares_stdrand
FULL=olmoe3_275m_emo_10b; RP=olmoe3_275m_emorand_square; HR=runs_heldout300b_emorand; HRB=runs_heldout300b_emo; START=emo_step19074; SAMPLE=sample_8k_300b.npz
B1RUNS="olmoe3_275m_emo_20b olmoe3_275m_emo_20b_filler olmoe3_275m_emo_20b_1node"; B2RUN=olmoe3_275m_emo_30b_1node
W1_STEPS=(20000 25000 30000 35000 38148); W2_STEPS=(39073 44073 49073 54073 57221)
LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRB
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
heldout() { local hr=$1 tag=$2 ckpt=$3; have $S/olmoe3_routing/$hr/$tag/none || launch "$SQN-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
steps_of() { python -c "import json; t=json.load(open('$1'))['tokens_per_group'][$2]; s=t//524288; print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')"; }
# ---- 0. same random groups + same random packs as the standard control; EMO start model sliced ----
[ -f $SQ/groups.json ] || cp $S/$SRC/groups.json $SQ/groups.json
[ -e $SQ/pack ] || ln -s ../$SRC/pack $SQ/pack; [ -e $SQ/pack2 ] || ln -s ../$SRC/pack2 $SQ/pack2
for g in 0 1 2 3; do [ -f $SQ/init/group$g/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g 2>&1 | tail -1; say "slice group$g done"; }; done
SHARES1=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
SHARES2=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack2/stats.json'))['token_share']))")
# ---- 1. EMO baseline 20B -> 30B; start model + window-1 baseline points on the 300B sample ----
launch_train $LOG/baseline30_launched $LOG/launch_baseline30.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_30b_baseline.sh
heldout $HRB $START $W/$FULL/step19074
for s in "${W1_STEPS[@]}"; do ck=""; for run in $B1RUNS; do for st in $s $((s-1)); do [ -f $S/$run/step$st/train/rank0.pt ] && { ck=$W/$run/step$st; break 2; }; done; done
  [ -n "$ck" ] && heldout $HRB baseline_step$s $ck || say "baseline checkpoint for step $s not found (skipped)"; done
# ---- 2. window-1 squares ----
declare -a S1 S2
for g in 0 1 2 3; do S1[$g]=$(steps_of $SQ/pack/stats.json $g)
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square
done
say "window-1 square steps: g0 [${S1[0]}] g1 [${S1[1]}] g2 [${S1[2]}] g3 [${S1[3]}]; shares $SHARES1"
for i in 0 1 2 3 4; do s=${W1_STEPS[$i]}
  [ -f $SQ/merged/match$s/merge_info.json ] && [ -f $SQ/ppl_validation/merged/match$s.json ] || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES1 SKIP_ORACLE=1 SAMPLE=$SAMPLE \
    STEPS_G0="${S1[0]}" STEPS_G1="${S1[1]}" STEPS_G2="${S1[2]}" STEPS_G3="${S1[3]}" bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
done
for i in 0 1 2 3 4; do s=${W1_STEPS[$i]}; for g in 0 1 2 3; do st=(${S1[$g]}); until [ -f $S/${RP}$g/step${st[$i]}/train/rank0.pt ]; do sleep 300; done; heldout $HR sub${g}_match$s $W/${RP}$g/step${st[$i]}; done; done
wait; say "window 1 merged"
# ---- 3. window 2: finals rewritten -> squares on pack2 -> merges + evals; baseline window-2 points ----
for g in 0 1 2 3; do st=(${S1[$g]}); S2[$g]=$(steps_of $SQ/pack2/stats.json $g)
  [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}$g/step${st[4]} --out $SQ/init2/group$g --overwrite 2>&1 | tail -1; say "init2 group$g done"; }
  tokens=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g])")
  launch_train $LOG/square${g}_w2_launched $LOG/launch_square${g}_w2.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_TOKENS=$tokens OLMOE3_DATA_PATHS="$W/$SQN/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$W/$SQN/init2/group$g/model_and_optim" OLMOE3_RUNNAME=${RP}${g}_w2 OLMOE3_WANDB_TAGS=$SQN,square,w2
done
say "window-2 square steps: g0 [${S2[0]}] g1 [${S2[1]}] g2 [${S2[2]}] g3 [${S2[3]}]; shares $SHARES2"
for i in 0 1 2 3 4; do s=${W2_STEPS[$i]}; subs=(); for g in 0 1 2 3; do st=(${S2[$g]}); subs+=("$S/${RP}${g}_w2/step${st[$i]}"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; done; done; say "window-2 point $s: square checkpoints present"
  [ -f $SQ/merged/match$s/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SHARES2" --out $SQ/merged/match$s --overwrite 2>&1 | tail -1 | cut -c1-120
  heldout $HR merged_match$s $W/$SQN/merged/match$s; ppl $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in 0 1 2 3; do st=(${S2[$g]}); heldout $HR sub${g}_match$s $W/${RP}${g}_w2/step${st[$i]}; done
  until [ -f $S/$B2RUN/step$s/train/rank0.pt ]; do sleep 600; done; heldout $HRB baseline_step$s $W/$B2RUN/step$s; ppl $W/$B2RUN/step$s $B2RUN/step$s.json baseline-$s
done
# ---- 4. collect ----
expected="$HRB/$START"; for s in "${W1_STEPS[@]}" "${W2_STEPS[@]}"; do expected="$expected $HR/merged_match$s"; [ $s = 20000 ] || expected="$expected $HRB/baseline_step$s"; for g in 0 1 2 3; do expected="$expected $HR/sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$tag/none; do sleep 300; done; done
for d in $S/olmoe3_routing/$HR/*/none $S/olmoe3_routing/$HRB/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "EMO random-partition control done"
