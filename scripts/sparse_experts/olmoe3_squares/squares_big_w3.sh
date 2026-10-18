#!/usr/bin/env bash
# Window 3 to the 61B point for the four-square LR 4e-4 controls (user request 2026-10-18): EMO 512e (olmoe3_squares_emorand_lr4e-4), EMO 1000e
# (olmoe3_squares_emo1000rand4) and EMO 2000e (olmoe3_squares_emo2000rand4). As in squares_control_w3.sh, each square continues from its window-2
# final on a CONTIGUOUS quarter of the stream after 30B (stream steps 57,221 + g x 47,684 -> ...), read through a finetune-start checkpoint
# (make_finetune_start.py pins the data position), at sub-model LR 4e-4; but only up to the 61B matched point: 14,815 steps per square
# (31.07% of the 100B window = the fraction the joint baseline has covered at step 116,479), one checkpoint at the end. Merged with equal weights
# (equal quarters) at matched step 116,479 and evaluated on the 300B held-out sample and the ppl sets (merged, every square alone, baseline).
# The 512e baseline point exists (olmoe3_275m_emo_130b/step116479); the 1000e / 2000e baselines are continued 30B -> 61B here (4 nodes, keeper).
# Running the squares further later (to 130B) only needs a relaunch with a larger FT_STEPS: the trainer resumes from the run folder.
#   bash scripts/sparse_experts/olmoe3_squares/squares_big_w3.sh emo512|emo1000|emo2000      (idempotent; detach it; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
ARM="${1:?emo512|emo1000|emo2000}"
case $ARM in
  emo512)  SQN=olmoe3_squares_emorand_lr4e-4; RP=olmoe3_275m_emorand_lr4e-4_square; FULL=olmoe3_275m_emo_10b;       E=512;  HR=runs_heldout300b_emorand_lr4e-4; HRB=runs_heldout300b_emo;     B30=olmoe3_275m_emo_30b_1node;  BRUN=olmoe3_275m_emo_130b;       BSCRIPT="";                                                                       SQENV="";                                       BSF=8; PPLB=8; BSQ=8;;
  emo1000) SQN=olmoe3_squares_emo1000rand4;    RP=olmoe3_275m_emo1000rand4_square;    FULL=olmoe3_275m_1000e_emo_10b; E=1000; HR=runs_heldout300b_emo1000rand4;    HRB=runs_heldout300b_emo1000; B30=olmoe3_275m_1000e_emo_30b;  BRUN=olmoe3_275m_1000e_emo_61b; BSCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_emo_61b_baseline.sh; SQENV="";                                       BSF=4; PPLB=8; BSQ=8;;
  emo2000) SQN=olmoe3_squares_emo2000rand4;    RP=olmoe3_275m_emo2000rand4_square;    FULL=olmoe3_275m_2000e_emo_10b; E=2000; HR=runs_heldout300b_emo2000rand4;    HRB=runs_heldout300b_emo2000; B30=olmoe3_275m_2000e_emo_30b;  BRUN=olmoe3_275m_2000e_emo_61b; BSCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_61b_baseline.sh; SQENV="OLMOE3_EP_SIZE=2 OLMOE3_EP_PATH=sync_1d"; BSF=2; PPLB=4; BSQ=2;;
  *) echo "emo512|emo1000|emo2000"; exit 1;;
esac
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; K=4; LR=4e-4
B0=57221; SQ_STEPS=47684; FT_STEPS=14815; POINT=116479   # 14,815 = round(47,684 x 5,926 / 19,074): the 61B fraction of each square's 100B-window quarter
SQ=$S/$SQN; LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRB $S/olmoe3_routing/baseline_keeper
say() { echo "$(date -u +%m-%d\ %H:%M) [w3 $ARM] $*"; }
LIVE=""; for attempt in 1 2 3 4 5; do LIVE=$(timeout 600 beaker workspace experiments ai2/flex2 --format=json 2>/dev/null | python -c "
import json,sys
try: es=json.load(sys.stdin)
except Exception: es=[]
for e in es:
    js=e.get('jobs') or []; st=(js[-1] if js else {}).get('status',{})
    if not st.get('exited') and not st.get('finalized') and not st.get('canceled'): print(e.get('name',''))"); [ -n "$LIVE" ] && break; sleep 60; done
[ -n "$LIVE" ] || { say "ABORT: could not list the live Beaker experiments"; exit 1; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  if grep -q -x -E "$(printf '%s' "$name" | sed 's/[.[\*^$]/\\&/g')(-[0-9a-f]+)*" <<< "$LIVE"; then say "$name: already live, not resubmitted"; return 0; fi
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local hr=$1 tag=$2 ckpt=$3 name=$4 bs=$5; have $S/olmoe3_routing/$hr/$tag/none || launch "$name" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size $bs --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size $PPLB; }
# ---- 1. baseline 30B -> 61B where it does not exist yet (4 nodes; keeper resumes it) ----
if [ -n "$BSCRIPT" ] && [ ! -f $S/$BRUN/step$POINT/train/rank0.pt ]; then launch_train $LOG/baseline61_launched $LOG/launch_baseline61.log $BSCRIPT
  pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh" > /dev/null || { ( setsid nohup bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh >> $S/olmoe3_routing/baseline_keeper/log.txt 2>&1 < /dev/null & ); say "baseline keeper started"; }; fi
# ---- 2. squares: window-2 finals rewritten -> finetune-start at each square's stream position -> 14,815 steps at LR 4e-4 ----
declare -a W2F ST
for g in 0 1 2 3; do W2F[$g]=$(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g] // 524288)"); done
pids=""; for g in 0 1 2 3; do [ -f $SQ/init3/group$g/model_and_optim/.metadata ] || { ( PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}${g}_w2/step${W2F[$g]} --out $SQ/init3/group$g --overwrite > $LOG/rewrite3_group$g.log 2>&1 && say "init3 group$g done" || say "init3 group$g FAILED" ) & pids="$pids $!"; }; done; [ -z "$pids" ] || wait $pids
for g in 0 1 2 3; do [ -f $SQ/init3/group$g/model_and_optim/.metadata ] || { say "ABORT: init3 group$g missing"; exit 1; }
  start=$((B0 + g * SQ_STEPS)); ST[$g]=$((start + FT_STEPS))
  [ -f $SQ/ft_start/w3_group$g/step$start/train/rank0.pt ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/init3/group$g --train-from $S/$B30/step57221 --start-step $start --out $SQ/ft_start/w3_group$g 2>&1 | tail -1 | cut -c1-120; say "ft_start group$g at step $start"; }
  launch_train $LOG/square${g}_w3_launched $LOG/launch_square${g}_w3.log scripts/sparse_experts/model_scripts/olmoe3_275m_stdrand_square_w3.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=1 OLMOE3_NUM_EXPERTS=$E OLMOE3_LR=$LR FT_START=$W/$SQN/ft_start/w3_group$g FT_START_STEP=$start FT_STEPS=$FT_STEPS OLMOE3_FIXED_STEPS=${ST[$g]} OLMOE3_WANDB_TAGS=$SQN,square,w3,lr4e-4 $SQENV
done
say "window-3 (61B point) square steps: $(for g in 0 1 2 3; do echo -n "g$g $((B0 + g * SQ_STEPS)) -> ${ST[$g]} "; done)"
# ---- 3. the 61B point: merge (equal quarters) + evaluations; baseline point when its checkpoint exists ----
basept() { [ -f $S/$BRUN/step$POINT/train/rank0.pt ] || return 0; heldout $HRB baseline_step$POINT $W/$BRUN/step$POINT "$ARM-eval-baseline_step$POINT" $BSF; ppl $W/$BRUN/step$POINT $BRUN/step$POINT.json "$ARM-ppl-baseline-$POINT"; }
basept
subs=(); for g in 0 1 2 3; do until [ -f $S/${RP}${g}_w3/step${ST[$g]}/train/rank0.pt ]; do sleep 300; basept; done; subs+=("$S/${RP}${g}_w3/step${ST[$g]}"); done; say "61B point: square checkpoints present"
[ -f $SQ/merged/match$POINT/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights 0.25,0.25,0.25,0.25 --out $SQ/merged/match$POINT --overwrite 2>&1 | tail -1 | cut -c1-120
heldout $HR merged_match$POINT $W/$SQN/merged/match$POINT "$SQN-eval-merged_match$POINT" $BSF; ppl $W/$SQN/merged/match$POINT merged/match$POINT.json "$SQN-ppl-merged-$POINT"
cmds=""; for g in 0 1 2 3; do have $S/olmoe3_routing/$HR/sub${g}_match$POINT/none || cmds="$cmds python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/${RP}${g}_w3/step${ST[$g]} --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/sub${g}_match$POINT/none/rank0 --restrict none --batch-size $BSQ --log-every 100;"; done
[ -n "$cmds" ] && launch "$SQN-squares-heldout-$POINT-h0" bash -c "$cmds"
ckpts=""; for g in 0 1 2 3; do [ -f $SQ/ppl_validation/${RP}${g}_w3/step${ST[$g]}.json ] || ckpts="$ckpts $W/${RP}${g}_w3/step${ST[$g]}"; done
[ -n "$ckpts" ] && launch "$SQN-squares-ppl-$POINT" python scripts/debug_validation/eval_ppl_validation.py --checkpoints $ckpts --out-dir "$W/$SQN/ppl_validation" --batch-size $PPLB
# ---- 4. collect ----
until [ -f $S/$BRUN/step$POINT/train/rank0.pt ]; do sleep 600; done; basept
for tag in $HR/merged_match$POINT $HRB/baseline_step$POINT $HR/sub0_match$POINT $HR/sub1_match$POINT $HR/sub2_match$POINT $HR/sub3_match$POINT; do until have $S/olmoe3_routing/$tag/none; do sleep 300; done; done
for d in $S/olmoe3_routing/$HR/*/none $S/olmoe3_routing/$HRB/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
