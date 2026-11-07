#!/usr/bin/env bash
# First-steps probe (user request 2026-11-07): is the merge degradation of the random squares immediate? EMO 512e from the 10B checkpoint at
# sub-model LR 4e-4, the existing random expert groups and document packs of the LR-4e-4 control arms, K = 8 squares for the first 10 steps
# (or K = 4 for the first 5): the squares checkpoint after every step (steps_worker.sh, allocated 8-GPU jobs through the CLI launcher), the
# driver merges the K squares after every step (token-share weights, as the control merges) and evaluates every merge on the reporting
# held-out sample (extract_routing.py --ce-only, batched allocated 1-GPU jobs). The step checkpoints and merged weights are deleted once their
# evaluation exists (user: they need not be kept); merge_info.json and the evaluation outputs stay.
#   bash scripts/sparse_experts/olmoe3_squares/squares_firststeps.sh k8|k4      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k4|k8}"; LR=4e-4
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_10b; SAMPLE=sample_8k_300b.npz
case $KV in k8) K=8; N=10; PARENT=olmoe3_squares_emorand8_lr4e-4; SQN=olmoe3_squares_emorand8_lr4e-4_steps; RP=olmoe3_275m_emorand8_lr4e-4_steps_square; HR=runs_heldout300b_emorand8_lr4e-4_steps; JOBS="0,1,2,3 4,5,6,7";;
            k4) K=4; N=5;  PARENT=olmoe3_squares_emorand_lr4e-4;  SQN=olmoe3_squares_emorand_lr4e-4_steps;  RP=olmoe3_275m_emorand_lr4e-4_steps_square;  HR=runs_heldout300b_emorand_lr4e-4_steps;  JOBS="0,1,2,3";; *) echo "k4|k8"; exit 1;; esac
SQ=$S/$SQN; LOG=$SQ/logs; mkdir -p $SQ $LOG $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [steps $KV] $*"; }
[ -f $SQ/groups.json ] || cp $S/$PARENT/groups.json $SQ/groups.json
for d in pack init; do [ -e $SQ/$d ] || ln -s ../$PARENT/$d $SQ/$d; done
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice $SQ/init/group$g missing"; exit 1; }; done
launch_job() { local name=$1 groups=$2 mark=$LOG/worker_$1.launched; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_LR=$LR STEPS=$N OLMOE3_WANDB_TAGS=$SQN,square,first_steps bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh ${groups//,/ }" 2>&1 | tail -n 1)
  say "$name: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
launch_eval() { local name=$1 cmds=$2 mark=$LOG/eval_$1.launched; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name" --gpus 1 -- bash -c "$cmds" 2>&1 | tail -n 1); say "$name: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
i=0; for groups in $JOBS; do launch_job "sqsteps-$KV-j$i" "$groups"; i=$((i+1)); done
SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
merge_step() { local s=$1; [ -f $SQ/merged/step$s/merge_info.json ] && return 0; local subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$s"); done
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged/step$s --overwrite 2>&1 | tail -n 1 | cut -c1-120; say "merged step $s"; }
have_all() { local s=$1 g; for g in $(seq 0 $((K-1))); do [ -f $S/${RP}$g/step$s/train/rank0.pt ] || return 1; done; }
until have_all $N; do sleep 120; done; say "all $K squares have their $N step checkpoints"
for s in $(seq 1 $N); do merge_step $s & [ $((s % 2)) = 0 ] && wait; done; wait; say "all merges done"
half=$(( (N + 1) / 2 )); cmds1=""; cmds2=""
for s in $(seq 1 $N); do c="python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$SQN/merged/step$s --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/merged_step$s/none --restrict none --batch-size 8 --log-every 100 --ce-only;"
  [ -f $S/olmoe3_routing/$HR/merged_step$s/none/meta.json ] && continue; [ $s -le $half ] && cmds1="$cmds1 $c" || cmds2="$cmds2 $c"; done
[ -n "$cmds1" ] && launch_eval "sqsteps-$KV-eval-a" "$cmds1"; [ -n "$cmds2" ] && launch_eval "sqsteps-$KV-eval-b" "$cmds2"
for s in $(seq 1 $N); do until [ -f $S/olmoe3_routing/$HR/merged_step$s/none/meta.json ]; do sleep 120; done; done; say "all evaluations done"
for s in $(seq 1 $N); do for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g/step$s"; done; rm -rf "$SQ/merged/step${s:?}/model_and_optim"; done   # user 2026-11-07: step checkpoints need not be kept
say "step checkpoints and merged weights deleted; results: $(for s in $(seq 1 $N); do echo -n "s$s=$(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/merged_step$s/none/meta.json'))['mean_ce'],4))") "; done)"; say done
