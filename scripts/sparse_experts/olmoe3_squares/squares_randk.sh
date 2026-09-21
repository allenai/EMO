#!/usr/bin/env bash
# Random-partition controls with K sub-models over the three windows (10B -> 20B -> 30B -> 130B), user request 2026-09-21:
#   std_k8   standard 512e, 8 random expert groups of 64, 8 random document groups (packs built here from the std assignment records)
#   s128_k4  standard 128e (same expert size, top-16 of 128), 4 groups of 32, the 512e control's random document packs
#   s128_k8  standard 128e, 8 groups of 16, the std_k8 packs
# Windows 1-2: document-level random packs (like the 4-square control); window 3: contiguous stream slices of 1/K of the 100B.
# Same matched checkpoints as the 512e control (20000 ... 38148 | 39073 ... 57221 | 66481 ... 247956). Baselines: the std ones exist;
# the 128e baseline (olmoe3_275m_128e_130b, 10B -> 130B jointly) is launched by s128_k4 and evaluated at every matched point.
#   bash scripts/sparse_experts/olmoe3_squares/squares_randk.sh std_k8|s128_k4|s128_k8   (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
V="${1:?std_k8|s128_k4|s128_k8}"; S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; SRCSTD=$S/olmoe3_squares_std
case $V in
  std_k8)  SQN=olmoe3_squares_stdrand8;  K=8; E=512; FULL=olmoe3_275m_10b;      RP=olmoe3_275m_stdrand8_square;  HR=runs_heldout300b_stdrand8;  HRB=runs_heldout300b_std;  START=std_step19074;  PACKS=build;                  BASE=""; BW1=olmoe3_275m_20b_1node; BW2=olmoe3_275m_30b_1node; BW3=olmoe3_275m_130b; SEED=2;;
  s128_k4) SQN=olmoe3_squares_s128rand4; K=4; E=128; FULL=olmoe3_275m_128e_10b; RP=olmoe3_275m_s128rand4_square; HR=runs_heldout300b_s128rand4; HRB=runs_heldout300b_s128; START=s128_step19074; PACKS=olmoe3_squares_stdrand;  BASE=olmoe3_275m_128e_130b_baseline.sh; BW1=olmoe3_275m_128e_130b; BW2=$BW1; BW3=$BW1; SEED=3;;
  s128_k8) SQN=olmoe3_squares_s128rand8; K=8; E=128; FULL=olmoe3_275m_128e_10b; RP=olmoe3_275m_s128rand8_square; HR=runs_heldout300b_s128rand8; HRB=runs_heldout300b_s128; START=s128_step19074; PACKS=olmoe3_squares_stdrand8; BASE="";                                 BW1=olmoe3_275m_128e_130b; BW2=$BW1; BW3=$BW1; SEED=4;;
  *) echo "std_k8|s128_k4|s128_k8"; exit 1;;
esac
SQ=$S/$SQN; W1=(20000 25000 30000 35000 38148); W2=(39073 44073 49073 54073 57221); W3=(66481 116479 166478 216477 247956); B0=57221; SQ3=$(( (190735 + K - 1) / K ))
LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRB
GS=$(seq 0 $((K-1)))
say() { echo "$(date -u +%m-%d\ %H:%M) [$V] $*"; }
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
shares_of() { python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$1'))['token_share']))"; }
merge_k() { local out=$1 weights=$2; shift 2; [ -f $out/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "$*")" --weights "$weights" --out $out --overwrite 2>&1 | tail -1 | cut -c1-120; }
# ---- 0. random expert groups; random document packs (built or shared); sliced start model; baseline; start-model evals ----
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k $K --seed $SEED --num-experts $E
if [ $PACKS = build ]; then
  [ -f $SQ/assign/records_7.npz ]  || python scripts/sparse_experts/olmoe3_squares/assign_random.py --src $SRCSTD/assign  --out $SQ/assign  --k $K --seed $SEED
  [ -f $SQ/assign2/records_7.npz ] || python scripts/sparse_experts/olmoe3_squares/assign_random.py --src $SRCSTD/assign2 --out $SQ/assign2 --k $K --seed $((SEED + 10))
  [ -f $SQ/pack/stats.json ]  || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign  --stream $S/olmoe3_routing/stream_10b_20b --out $SQ/pack  --groups $SQ/groups.json 2>&1 | tail -1; say "pack done"; }
  [ -f $SQ/pack2/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign2 --stream $S/olmoe3_routing/stream_20b_30b --out $SQ/pack2 --groups $SQ/groups.json 2>&1 | tail -1; say "pack2 done"; }
else
  [ -e $SQ/pack ] || ln -s ../$PACKS/pack $SQ/pack; [ -e $SQ/pack2 ] || ln -s ../$PACKS/pack2 $SQ/pack2
  until [ -f $SQ/pack/stats.json ] && [ -f $SQ/pack2/stats.json ]; do sleep 300; done
fi
for g in $GS; do [ -f $SQ/init/group$g/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g 2>&1 | tail -1; say "slice group$g done"; }; done
if [ -n "$BASE" ]; then
  launch_train $LOG/baseline_launched $LOG/launch_baseline.log scripts/sparse_experts/model_scripts/$BASE
  heldout $HRB $START $W/$FULL/step19074
  [ -f claude_outputs/debug_validation/ppl_validation/$FULL/step19074.json ] || launch "$SQN-ppl-start" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$FULL/step19074" --out-dir "/weka/oe-training-default/ryanwang/EMO/claude_outputs/debug_validation/ppl_validation" --batch-size 8
fi
SH1=$(shares_of $SQ/pack/stats.json); SH2=$(shares_of $SQ/pack2/stats.json); SH3=$(python -c "print(','.join(['%.6f' % (1.0/$K)] * $K))")
# ---- 1. window 1 ----
declare -a S1 S2 S3
for g in $GS; do S1[$g]=$(steps_of $SQ/pack/stats.json $g)
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_EMO=0 OLMOE3_NUM_EXPERTS=$E OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square
done
say "window-1 steps: $(for g in $GS; do echo -n "g$g [${S1[$g]}] "; done); shares $SH1"
# (merge_point.sh takes STEPS_G<g> as space-separated lists; export them explicitly)
for g in $GS; do export STEPS_G$g="${S1[$g]}"; done
for i in 0 1 2 3 4; do s=${W1[$i]}
  [ -f $SQ/merged/match$s/merge_info.json ] && [ -f $SQ/ppl_validation/merged/match$s.json ] || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SH1 SKIP_ORACLE=1 SAMPLE=$SAMPLE K=$K bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
done
for i in 0 1 2 3 4; do s=${W1[$i]}; for g in $GS; do st=(${S1[$g]}); until [ -f $S/${RP}$g/step${st[$i]}/train/rank0.pt ]; do sleep 300; done; heldout $HR sub${g}_match$s $W/${RP}$g/step${st[$i]}; done
  [ -n "$BASE" ] && { bs=$s; [ $s = 38148 ] && [ ! -f $S/$BW1/step$s/train/rank0.pt ] && bs=38147; until [ -f $S/$BW1/step$bs/train/rank0.pt ]; do sleep 600; done; heldout $HRB baseline_step$s $W/$BW1/step$bs; ppl $W/$BW1/step$bs $BW1/step$bs.json baseline-$s; }
done
wait; say "window 1 merged"
# ---- 2. window 2 ----
for g in $GS; do st=(${S1[$g]}); S2[$g]=$(steps_of $SQ/pack2/stats.json $g)
  [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}$g/step${st[4]} --out $SQ/init2/group$g --overwrite 2>&1 | tail -1; say "init2 group$g done"; }
  tokens=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g])")
  launch_train $LOG/square${g}_w2_launched $LOG/launch_square${g}_w2.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_EMO=0 OLMOE3_NUM_EXPERTS=$E OLMOE3_TOKENS=$tokens OLMOE3_DATA_PATHS="$W/$SQN/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$W/$SQN/init2/group$g/model_and_optim" OLMOE3_RUNNAME=${RP}${g}_w2 OLMOE3_WANDB_TAGS=$SQN,square,w2
done
for i in 0 1 2 3 4; do s=${W2[$i]}; subs=(); for g in $GS; do st=(${S2[$g]}); subs+=("$S/${RP}${g}_w2/step${st[$i]}"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; done; done; say "window-2 point $s: square checkpoints present"
  merge_k $SQ/merged/match$s "$SH2" "${subs[@]}"; heldout $HR merged_match$s $W/$SQN/merged/match$s; ppl $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in $GS; do st=(${S2[$g]}); heldout $HR sub${g}_match$s $W/${RP}${g}_w2/step${st[$i]}; done
  [ -n "$BASE" ] && { until [ -f $S/$BW2/step$s/train/rank0.pt ]; do sleep 600; done; heldout $HRB baseline_step$s $W/$BW2/step$s; ppl $W/$BW2/step$s $BW2/step$s.json baseline-$s; }
done
# ---- 3. window 3: contiguous stream slices of 1/K of the 100B ----
for g in $GS; do st=(${S2[$g]}); start=$((B0 + g * SQ3))
  [ -f $SQ/init3/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}${g}_w2/step${st[4]} --out $SQ/init3/group$g --overwrite 2>&1 | tail -1; say "init3 group$g done"; }
  [ -f $SQ/ft_start/w3_group$g/step$start/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/init3/group$g --train-from $S/olmoe3_275m_30b_1node/step57221 --start-step $start --out $SQ/ft_start/w3_group$g 2>&1 | tail -1
  S3[$g]=$(python -c "s=$SQ3; b=$start; print(' '.join(str(b + max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {b + s}')")
  launch_train $LOG/square${g}_w3_launched $LOG/launch_square${g}_w3.log scripts/sparse_experts/model_scripts/olmoe3_275m_stdrand_square_w3.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=0 OLMOE3_NUM_EXPERTS=$E FT_START=$W/$SQN/ft_start/w3_group$g FT_START_STEP=$start FT_STEPS=$SQ3
done
say "window-3 steps: $(for g in $GS; do echo -n "g$g [${S3[$g]}] "; done)"
for i in 0 1 2 3 4; do s=${W3[$i]}; subs=(); for g in $GS; do st=(${S3[$g]}); subs+=("$S/${RP}${g}_w3/step${st[$i]}"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 600; done; done; say "window-3 point $s: square checkpoints present"
  merge_k $SQ/merged/match$s "$SH3" "${subs[@]}"; heldout $HR merged_match$s $W/$SQN/merged/match$s; ppl $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in $GS; do st=(${S3[$g]}); heldout $HR sub${g}_match$s $W/${RP}${g}_w3/step${st[$i]}; done
  [ -n "$BASE" ] && { until [ -f $S/$BW3/step$s/train/rank0.pt ]; do sleep 600; done; heldout $HRB baseline_step$s $W/$BW3/step$s; ppl $W/$BW3/step$s $BW3/step$s.json baseline-$s; }
done
# ---- 4. collect ----
expected=""; for s in "${W1[@]}" "${W2[@]}" "${W3[@]}"; do expected="$expected $HR/merged_match$s"; for g in $GS; do expected="$expected $HR/sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$tag/none; do sleep 300; done; done
for d in $S/olmoe3_routing/$HR/*/none $S/olmoe3_routing/$HRB/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
