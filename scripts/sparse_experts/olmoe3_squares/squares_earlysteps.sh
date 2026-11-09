#!/usr/bin/env bash
# Early-steps probe, continued (user request 2026-11-09): the first-steps probe (squares_firststeps.sh) showed the 8-square merge jumping to its
# 15.7B level within ~7 steps and the 4-square merge rising for 5 steps; the next reading was at 15.7B (1,365 / 2,731 square steps). This run
# fills the gap up to 300 square steps with log-spaced checkpoints (15 20 30 50 70 100 150 200 300): the same start slices, expert groups,
# document packs, data order and constant LR 4e-4 as the probe and the window-1 runs (exact prefixes of the same schedule), one 8-GPU allocated
# job per 4 squares (steps_worker.sh with CKPT_STEPS), merge after every checkpoint (token-share weights), evaluate every merge AND square 0
# alone on the reporting held-out sample (--ce-only) in ONE 8-GPU job (eval_batch_8gpu.sh), then delete the step checkpoints and merged weights.
#   bash scripts/sparse_experts/olmoe3_squares/squares_earlysteps.sh k8|k4      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k4|k8}"; LR=4e-4; STEPS="${STEPS:-15 20 30 50 70 100 150 200 300}"; N=${STEPS##* }
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_10b; SAMPLE=sample_8k_300b.npz
case $KV in k8) K=8; PARENT=olmoe3_squares_emorand8_lr4e-4; SQN=olmoe3_squares_emorand8_lr4e-4_steps300; RP=olmoe3_275m_emorand8_lr4e-4_steps300_square; HR=runs_heldout300b_emorand8_lr4e-4_steps300; JOBS="0,1,2,3 4,5,6,7";;
            k4) K=4; PARENT=olmoe3_squares_emorand_lr4e-4;  SQN=olmoe3_squares_emorand_lr4e-4_steps300;  RP=olmoe3_275m_emorand_lr4e-4_steps300_square;  HR=runs_heldout300b_emorand_lr4e-4_steps300;  JOBS="0,1,2,3";; *) echo "k4|k8"; exit 1;; esac
SQ=$S/$SQN; LOG=$SQ/logs; mkdir -p $SQ $LOG $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [steps300 $KV] $*"; }
[ -f $SQ/groups.json ] || cp $S/$PARENT/groups.json $SQ/groups.json
for d in pack init; do [ -e $SQ/$d ] || ln -s ../$PARENT/$d $SQ/$d; done
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice $SQ/init/group$g missing"; exit 1; }; done
launch_job() { local name=$1 groups=$2 mark=$LOG/worker_$1.launched; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_LR=$LR STEPS=$N CKPT_STEPS=${STEPS// /,} OLMOE3_WANDB_TAGS=$SQN,square,early_steps bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh ${groups//,/ }" 2>&1 | tail -n 1)
  say "$name: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
i=0; for groups in $JOBS; do launch_job "sqsteps300-$KV-j$i" "$groups"; i=$((i+1)); done
SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
merge_step() { local s=$1; [ -f $SQ/merged/step$s/merge_info.json ] && return 0; local subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$s"); done
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged/step$s --overwrite 2>&1 | tail -n 1 | cut -c1-120; say "merged step $s"; }
have_all() { local s=$1 g; for g in $(seq 0 $((K-1))); do [ -f $S/${RP}$g/step$s/train/rank0.pt ] || return 1; done; }
until have_all $N; do sleep 120; done; say "all $K squares have their step-$N checkpoints"
n=0; for s in $STEPS; do merge_step $s & n=$((n+1)); [ $((n % 2)) = 0 ] && wait; done; wait; say "all merges done"
# one 8-GPU evaluation job: every merge and square 0 alone at every step, reporting held-out sample, CE only
CF=$LOG/evals_8gpu.txt; : > $CF
ev() { local name=$1 ckpt=$2; [ -f $S/olmoe3_routing/$HR/$name/none/meta.json ] || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $ckpt --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$name/none --restrict none --batch-size 8 --log-every 100 --ce-only" >> $CF; }
for s in $STEPS; do ev merged_step$s $W/$SQN/merged/step$s; ev sub0_step$s $W/${RP}0/step$s; done
if [ -s $CF ] && [ ! -f $LOG/eval8gpu.launched ]; then u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "sqsteps300-$KV-evals" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_8gpu.txt 2>&1 | tail -n 1)
  say "evals ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/eval8gpu.launched; fi
for s in $STEPS; do for p in merged_step$s sub0_step$s; do until [ -f $S/olmoe3_routing/$HR/$p/none/meta.json ]; do sleep 120; done; done; done; say "all evaluations done"
for s in $STEPS; do for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g/step${s:?}"; done; rm -rf "$SQ/merged/step${s:?}/model_and_optim"; done   # as the probe: step checkpoints need not be kept
say "step checkpoints and merged weights deleted; merged: $(for s in $STEPS; do echo -n "s$s=$(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/merged_step$s/none/meta.json'))['mean_ce'],4))") "; done)"; say done
