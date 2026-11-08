#!/usr/bin/env bash
# Window 3 beyond 61B for the EMO 2000e four-square LR-4e-4 control (user request 2026-11-08): the four w3 squares resume from their 61B
# checkpoints (the trainer prefers the run folder's latest checkpoint over the finetune-start load path) and run to the end of their
# contiguous quarter of the 100B window, with checkpoints at the squares' 87B / 113B / 130B fractions (27,315 / 39,815 / 47,684 of 47,684
# steps); the joint baseline is continued 61B -> 130B (olmoe3_275m_2000e_emo_130b_baseline.sh, 4 nodes, keeper). At every point: equal-weight
# merge, then ONE 8-GPU evaluation job per point (eval_batch_8gpu.sh: merged + every square + baseline, held-out sample and ppl sets).
# GATE (user: the running analysis must not be blocked): nothing is submitted until the experiments listed in GATE_IDS have started.
#   bash scripts/sparse_experts/olmoe3_squares/squares_big_w3_130b.sh emo2000      (idempotent; detach it; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
ARM="${1:?emo2000}"; [ $ARM = emo2000 ] || { echo "emo2000 only"; exit 1; }
SQN=olmoe3_squares_emo2000rand4; RP=olmoe3_275m_emo2000rand4_square; FULL=olmoe3_275m_2000e_emo_10b; E=2000; HR=runs_heldout300b_emo2000rand4; HRB=runs_heldout300b_emo2000
BRUN=olmoe3_275m_2000e_emo_130b; BSCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_130b_baseline.sh; SQENV="OLMOE3_EP_SIZE=2 OLMOE3_EP_PATH=sync_1d"; BSF=2; PPLB=4
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; LR=4e-4
B0=57221; SQ_STEPS=47684; DONE_STEPS=14815; POINTS="166478:27315 216477:39815 247956:47684"   # matched step : the square's step count at that fraction of its quarter
SQ=$S/$SQN; LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $LOG $SP $S/olmoe3_routing/baseline_keeper
GATE_IDS="${GATE_IDS:-01M6WR2TEK3A8QKXVKVXFVPY0E $(grep -o '[A-Z0-9]*$' $S/olmoe3_squares_emo2000lb/logs/lb1e-1_launched 2>/dev/null)}"   # the consolidated evaluation job + the last LB-sweep arm
say() { echo "$(date -u +%m-%d\ %H:%M) [w3-130b $ARM] $*"; }
state() { beaker experiment get "$1" --format=json 2>/dev/null | python -c "
import json,sys
try: d=json.load(sys.stdin)[0]; s=(d.get('jobs') or [{}])[-1].get('status',{}); k=[x for x in ('created','scheduled','started','exited','canceled','finalized') if s.get(x)]; print(k[-1] if k else 'none')
except Exception: print('unknown')"; }
for id in $GATE_IDS; do until [ "$(state $id)" != created ] && [ "$(state $id)" != scheduled ] && [ "$(state $id)" != unknown ]; do sleep 300; done; say "gate: $id is $(state $id)"; done
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
# ---- 1. baseline 61B -> 130B (4 nodes; keeper resumes it) ----
[ -f $S/$BRUN/step247956/train/rank0.pt ] || { launch_train $LOG/baseline130_launched $LOG/launch_baseline130.log $BSCRIPT
  pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh" > /dev/null || { ( setsid nohup bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh >> $S/olmoe3_routing/baseline_keeper/log.txt 2>&1 < /dev/null & ); say "baseline keeper started"; }; }
# ---- 2. squares: resume from the 61B checkpoints to the end of the quarter ----
for g in 0 1 2 3; do start=$((B0 + g * SQ_STEPS)); fixed=$(for p in $POINTS; do echo -n "$((start + ${p#*:})),"; done); fixed=${fixed%,}
  [ -f $S/${RP}${g}_w3/step$((start + DONE_STEPS))/train/rank0.pt ] || { say "ABORT: square $g has no 61B checkpoint"; exit 1; }
  [ -f $S/${RP}${g}_w3/step$((start + SQ_STEPS))/train/rank0.pt ] && continue
  launch_train $LOG/square${g}_w3b_launched $LOG/launch_square${g}_w3b.log scripts/sparse_experts/model_scripts/olmoe3_275m_stdrand_square_w3.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=1 OLMOE3_NUM_EXPERTS=$E OLMOE3_LR=$LR FT_START=$W/$SQN/ft_start/w3_group$g FT_START_STEP=$start FT_STEPS=$SQ_STEPS OLMOE3_FIXED_STEPS=$fixed OLMOE3_WANDB_TAGS=$SQN,square,w3,to130b $SQENV
done
say "squares relaunched to the end of their quarters; points: $POINTS"
# ---- 3. per point: merge when the squares are there, one 8-GPU evaluation job when the baseline point is there too ----
for p in $POINTS; do P=${p%%:*}; F=${p#*:}; subs=()
  for g in 0 1 2 3; do st=$((B0 + g * SQ_STEPS + F)); until [ -f $S/${RP}${g}_w3/step$st/train/rank0.pt ]; do sleep 600; done; subs+=("$S/${RP}${g}_w3/step$st"); done; say "point $P: square checkpoints present"
  [ -f $SQ/merged/match$P/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights 0.25,0.25,0.25,0.25 --out $SQ/merged/match$P --overwrite 2>&1 | tail -n 1 | cut -c1-120
  until [ -f $S/$BRUN/step$P/train/rank0.pt ]; do sleep 600; done; say "point $P: baseline checkpoint present"
  CF=$LOG/evals_$P.txt; : > $CF
  have $S/olmoe3_routing/$HR/merged_match$P/none || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$SQN/merged/match$P --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/merged_match$P/none/rank0 --restrict none --batch-size $BSF --log-every 100" >> $CF
  have $S/olmoe3_routing/$HRB/baseline_step$P/none || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$BRUN/step$P --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HRB/baseline_step$P/none/rank0 --restrict none --batch-size $BSF --log-every 100" >> $CF
  for g in 0 1 2 3; do st=$((B0 + g * SQ_STEPS + F)); have $S/olmoe3_routing/$HR/sub${g}_match$P/none || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/${RP}${g}_w3/step$st --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/sub${g}_match$P/none/rank0 --restrict none --batch-size $BSF --log-every 100" >> $CF; done
  [ -f $SQ/ppl_validation/merged/match$P.json ] || echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $W/$SQN/merged/match$P --out-dir $W/$SQN/ppl_validation --batch-size $PPLB" >> $CF
  [ -f $SQ/ppl_validation/$BRUN/step$P.json ] || echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $W/$BRUN/step$P --out-dir $W/$SQN/ppl_validation --batch-size $PPLB" >> $CF
  ck=""; for g in 0 1 2 3; do st=$((B0 + g * SQ_STEPS + F)); [ -f $SQ/ppl_validation/${RP}${g}_w3/step$st.json ] || ck="$ck $W/${RP}${g}_w3/step$st"; done; [ -n "$ck" ] && echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $ck --out-dir $W/$SQN/ppl_validation --batch-size $PPLB" >> $CF
  if [ -s $CF ] && [ ! -f $LOG/evals_$P.launched ]; then u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$SQN-evals-$P" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_$P.txt 2>&1 | tail -n 1); say "evals $P ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/evals_$P.launched; fi
done
for p in $POINTS; do P=${p%%:*}; for tag in $HR/merged_match$P $HRB/baseline_step$P $HR/sub0_match$P $HR/sub1_match$P $HR/sub2_match$P $HR/sub3_match$P; do until have $S/olmoe3_routing/$tag/none; do sleep 600; done; done; done
for d in $S/olmoe3_routing/$HR/*/none $S/olmoe3_routing/$HRB/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -n 1; done
say "done"
