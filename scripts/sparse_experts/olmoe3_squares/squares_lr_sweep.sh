#!/usr/bin/env bash
# Sub-model learning-rate sweep for the EMO 512e random-partition controls (Q5 box 5A; user request 2026-09-26). One arm = one LR shared by
# all K squares; everything else (random expert groups, random document packs, sliced start checkpoints, EMO loss, batch, constant LR
# schedule) is exactly the parent arm's (olmoe3_squares_emorand / emorand8). Window 1 (10B -> 20B): the squares checkpoint only at the
# mid-window matched point (baseline step 30000, 57% of the window) and at the window final (38148); both are merged and evaluated on
# (a) the SELECTION sample sample_8k_300b_val.npz (6,808 instances of the 300B stream slice, disjoint from the reporting sample),
# (b) the reporting sample sample_8k_300b.npz and (c) the v3-small ppl sets. The LR is chosen on (a) at 30000. Window 2 (20B -> 30B)
# is run for the chosen LR only (mode w2): final checkpoint 57221, merged, same three evaluations.
#   bash scripts/sparse_experts/olmoe3_squares/squares_lr_sweep.sh k4|k8 <lr> w1|w2      (idempotent; detach; commit + push first)
#   bash scripts/sparse_experts/olmoe3_squares/squares_lr_sweep.sh k4|k8 ref             (selection-sample passes of the parent 8e-4 arm's merges)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k4|k8}"; LR="${2:?lr, e.g. 2e-4}"; MODE="${3:-w1}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_10b; SAMPLE=sample_8k_300b.npz; VSAMPLE=sample_8k_300b_val.npz
case $KV in k4) K=4; PARENT=olmoe3_squares_emorand;  PRP=olmoe3_275m_emorand_square;  PHR=runs_heldout300b_emorand;;
            k8) K=8; PARENT=olmoe3_squares_emorand8; PRP=olmoe3_275m_emorand8_square; PHR=runs_heldout300b_emorand8;; *) echo "k4|k8"; exit 1;; esac
SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
say() { echo "$(date -u +%m-%d\ %H:%M) [lr $KV $LR $MODE] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local hr=$1 tag=$2 ckpt=$3 sample=$4 name=$5; have $S/olmoe3_routing/$hr/$tag/none || launch "$name" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$sample" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3 sqn=$4; [ -f $S/$sqn/ppl_validation/$json ] || launch "$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$sqn/ppl_validation" --batch-size 8; }
shares_of() { python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$1'))['token_share']))"; }
merge_k() { local sqn=$1 out=$2 weights=$3; shift 3; [ -f $out/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $S/$sqn/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "$*")" --weights "$weights" --out $out --overwrite 2>&1 | tail -1 | cut -c1-120; }
GS=$(seq 0 $((K-1)))
if [ $MODE = ref ]; then   # the parent arm (LR 8e-4) on the selection sample, at the same points
  for s in 30000 38148 57221; do heldout ${PHR}_val merged_match$s $W/$PARENT/merged/match$s $VSAMPLE "$PARENT-val-merged_match$s"; done
  say "ref passes launched"; exit 0
fi
TAG=lr$LR; SQN=${PARENT}_$TAG; RP=${PRP%_square}_${TAG}_square; HR=${PHR}_$TAG; HRV=${PHR}_${TAG}_val; SQ=$S/$SQN; LOG=$SQ/logs; mkdir -p $SQ $LOG $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRV
echo "$KV $LR" > $SQ/driver_args
[ -f $SQ/groups.json ] || cp $S/$PARENT/groups.json $SQ/groups.json
for d in pack pack2 init; do [ -e $SQ/$d ] || ln -s ../$PARENT/$d $SQ/$d; done
mid() { python -c "s=$1; print(max(1, round(s*10926/19074)))"; }
if [ $MODE = w1 ]; then
  SH1=$(shares_of $SQ/pack/stats.json); declare -a FIN MID
  for g in $GS; do t=$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g])"); FIN[$g]=$((t / 524288)); MID[$g]=$(mid ${FIN[$g]})
    launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_EMO=1 OLMOE3_NUM_EXPERTS=512 OLMOE3_LR=$LR OLMOE3_FIXED_STEPS="${MID[$g]},${FIN[$g]}" OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square,lr_sweep
  done
  say "window-1 steps: $(for g in $GS; do echo -n "g$g mid ${MID[$g]} final ${FIN[$g]} "; done); shares $SH1"
  for point in 30000:MID 38148:FIN; do s=${point%%:*}; arr=${point##*:}; subs=()
    for g in $GS; do eval "st=\${$arr[$g]}"; until [ -f $S/${RP}$g/step$st/train/rank0.pt ]; do sleep 300; done; subs+=("$S/${RP}$g/step$st"); done; say "point $s: square checkpoints present"
    merge_k $SQN $SQ/merged/match$s "$SH1" "${subs[@]}"
    heldout $HRV merged_match$s $W/$SQN/merged/match$s $VSAMPLE "$SQN-val-merged_match$s"
    heldout $HR merged_match$s $W/$SQN/merged/match$s $SAMPLE "$SQN-eval-merged_match$s"
    ppl $W/$SQN/merged/match$s merged/match$s.json "$SQN-ppl-merged-$s" $SQN
  done
  for s in 30000 38148; do for hr in $HRV $HR; do until have $S/olmoe3_routing/$hr/merged_match$s/none; do sleep 300; done; done; done
  say "done"
else   # w2: window 2 for the chosen LR, from this arm's window-1 finals
  SH2=$(shares_of $SQ/pack2/stats.json); subs=()
  for g in $GS; do t=$(python -c "import json; print(json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g])"); fin=$((t / 524288))
    [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}$g/step$fin --out $SQ/init2/group$g --overwrite 2>&1 | tail -1; say "init2 group$g done"; }
    t2=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g])"); fin2=$((t2 / 524288)); subs+=("$S/${RP}${g}_w2/step$fin2")
    launch_train $LOG/square${g}_w2_launched $LOG/launch_square${g}_w2.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_EMO=1 OLMOE3_NUM_EXPERTS=512 OLMOE3_LR=$LR OLMOE3_TOKENS=$t2 OLMOE3_FIXED_STEPS="$fin2" OLMOE3_DATA_PATHS="$W/$SQN/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$W/$SQN/init2/group$g/model_and_optim" OLMOE3_RUNNAME=${RP}${g}_w2 OLMOE3_WANDB_TAGS=$SQN,square,lr_sweep,w2
  done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; done; done; say "window-2 finals present"
  merge_k $SQN $SQ/merged/match57221 "$SH2" "${subs[@]}"
  heldout $HRV merged_match57221 $W/$SQN/merged/match57221 $VSAMPLE "$SQN-val-merged_match57221"
  heldout $HR merged_match57221 $W/$SQN/merged/match57221 $SAMPLE "$SQN-eval-merged_match57221"
  ppl $W/$SQN/merged/match57221 merged/match57221.json "$SQN-ppl-merged-57221" $SQN
  for hr in $HRV $HR; do until have $S/olmoe3_routing/$hr/merged_match57221/none; do sleep 300; done; done
  say "done"
fi
