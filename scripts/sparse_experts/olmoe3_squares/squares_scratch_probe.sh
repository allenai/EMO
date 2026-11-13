#!/usr/bin/env bash
# From-scratch squares, first-steps probe (user request 2026-11-13): does a standard-MoE 512e model train when it is split into 4 or 8
# random squares from the RANDOM INIT? Start model = step0 of olmoe3_275m_scratch_joint_steps800 (the recipe's random init, saved by
# OLMOE3_PRE_TRAIN_CKPT=1); random expert groups (seed 0) and the documents of the run's own first 10B tokens (stream_0b_10b) split
# uniformly at random into K packs. Each square trains 100 steps of its pack at peak LR 8e-4 with a linear warmup of 2000/K steps (the joint
# recipe's 2,000-step warmup in tokens: K squares together consume K batches per step), checkpoints at 10 20 30 50 70 100; every checkpoint
# is merged (token-share weights; untouched tensors copied) and scored on the reporting held-out sample (CE only), as are all squares alone.
# `joint` scores the joint from-scratch run at step 0 and its checkpoints (10 ... 800; 800 joint steps = the tokens of 8 squares x 100 steps).
# All evaluations of a mode run in ONE 8-GPU job; step checkpoints and merged weights are deleted afterwards (step0 and the packs are kept
# for the 10B run).
#   bash scripts/sparse_experts/olmoe3_squares/squares_scratch_probe.sh k4|k8|joint      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k4|k8|joint}"; LR="${LR:-8e-4}"; STEPS="10 20 30 50 70 100"; N=100; JSTEPS="0 10 20 30 50 70 100 150 200 300 400 600 800"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; JOINT=olmoe3_275m_scratch_joint_steps800; ST=$S/olmoe3_routing/stream_0b_10b
HR=runs_heldout300b_scratch_steps; mkdir -p $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [scratch probe $KV] $*"; }
ev() { local name=$1 ckpt=$2; [ -f $S/olmoe3_routing/$HR/$name/none/meta.json ] || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $ckpt --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$name/none --restrict none --batch-size 8 --log-every 100 --ce-only" >> $CF; }
run_evals() { local tag=$1; if [ -s $CF ] && [ ! -f $LOG/eval8gpu.launched ]; then local u; u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "scratchprobe-$tag-evals" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh ${CF/$S/$W} 2>&1 | tail -n 1)
  say "evals ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/eval8gpu.launched; fi; }
if [ $KV = joint ]; then LOG=$S/olmoe3_squares_scratch/logs/joint; mkdir -p $LOG; CF=$LOG/evals_8gpu.txt
  until [ -f $S/$JOINT/step800/train/rank0.pt ]; do sleep 120; done; say "joint run has step 800"
  : > $CF; for s in $JSTEPS; do ev joint_step$s $W/$JOINT/step$s; done; run_evals joint
  for s in $JSTEPS; do until [ -f $S/olmoe3_routing/$HR/joint_step$s/none/meta.json ]; do sleep 120; done; done
  for s in $JSTEPS; do [ $s = 0 ] || rm -rf "$S/${JOINT:?}/step${s:?}"; done
  say "joint: $(for s in $JSTEPS; do echo -n "s$s=$(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/joint_step$s/none/meta.json'))['mean_ce'],4))") "; done)"; say done; exit 0; fi
case $KV in k4) K=4;; k8) K=8;; *) echo "k4|k8|joint"; exit 1;; esac
SQN=olmoe3_squares_scratch$K; SQ=$S/$SQN; LOG=$SQ/logs; RP=olmoe3_275m_scratch${K}_steps100_square; WARM=$((2000 / K)); mkdir -p $SQ $LOG; CF=$LOG/evals_8gpu.txt
# ---- 0. groups, documents, packs, start slices ----
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k $K --seed 0
until [ -f $ST/manifest.json ]; do sleep 120; done
[ -f $SQ/assign/.done ] || { python scripts/sparse_experts/olmoe3_squares/assign_random_stream.py --stream $ST --out $SQ/assign --k $K --seed 0 2>&1 | tail -n 1 && touch $SQ/assign/.done; }
[ -f $SQ/pack/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign --stream $ST --out $SQ/pack --groups $SQ/groups.json > $LOG/pack.log 2>&1; say "pack: $(tail -n 1 $LOG/pack.log | cut -c1-160)"; }
[ -f $SQ/pack/stats.json ] || { say "ABORT: pack failed"; exit 1; }
until [ -f $S/$JOINT/step0/train/rank0.pt ]; do sleep 120; done
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] && continue
  { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=4 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$JOINT/step0 --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; say "slice group$g: $(tail -n 1 $LOG/slice_group$g.log | cut -c1-100)"; } & done; wait
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice $SQ/init/group$g missing"; exit 1; }; done
# ---- 1. the squares: one 8-GPU job, back to back ----
if [ ! -f $LOG/worker.launched ]; then
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "scratchprobe-k$K" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=0 OLMOE3_LR=$LR OLMOE3_WARMUP=$WARM STEPS=$N CKPT_STEPS=${STEPS// /,} OLMOE3_WANDB_TAGS=$SQN,square,first_steps,scratch bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh $(seq -s' ' 0 $((K-1)))" 2>&1 | tail -n 1)
  say "squares: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/worker.launched; fi
# ---- 2. merges and evaluations ----
SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
merge_step() { local s=$1; [ -f $SQ/merged/step$s/merge_info.json ] && return 0; local subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$s"); done
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$JOINT/step0 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged/step$s --overwrite 2>&1 | tail -n 1 | cut -c1-120; say "merged step $s"; }
have_all() { local s=$1 g; for g in $(seq 0 $((K-1))); do [ -f $S/${RP}$g/step$s/train/rank0.pt ] || return 1; done; }
until have_all $N; do sleep 120; done; say "all $K squares have their step-$N checkpoints"
n=0; for s in $STEPS; do merge_step $s & n=$((n+1)); [ $((n % 3)) = 0 ] && wait; done; wait; say "all merges done"
: > $CF; for s in $STEPS; do ev k${K}_merged_step$s $W/$SQN/merged/step$s; for g in $(seq 0 $((K-1))); do ev k${K}_sub${g}_step$s $W/${RP}$g/step$s; done; done; run_evals k$K
for s in $STEPS; do until [ -f $S/olmoe3_routing/$HR/k${K}_merged_step$s/none/meta.json ]; do sleep 120; done; for g in $(seq 0 $((K-1))); do until [ -f $S/olmoe3_routing/$HR/k${K}_sub${g}_step$s/none/meta.json ]; do sleep 120; done; done; done; say "all evaluations done"
for s in $STEPS; do for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g/step${s:?}"; done; rm -rf "$SQ/merged/step${s:?}/model_and_optim"; done
ce() { python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/$1/none/meta.json'))['mean_ce'],4))"; }
say "merged: $(for s in $STEPS; do echo -n "s$s=$(ce k${K}_merged_step$s) "; done)"; say "square 0: $(for s in $STEPS; do echo -n "s$s=$(ce k${K}_sub0_step$s) "; done)"; say done
