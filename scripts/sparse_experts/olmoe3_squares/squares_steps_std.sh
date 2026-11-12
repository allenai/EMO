#!/usr/bin/env bash
# First-steps probe for a second start model (user requests 2026-11-10 std / 2026-11-12 mixed-pool; the EMO version is squares_firststeps.sh +
# squares_earlysteps.sh): the model's random 4- and 8-square control arms' expert groups, document packs and start slices (re-sliced when the
# control arm's init/ is gone), the squares trained at sub-model LR 4e-4 with checkpoints after every one of the first 10 steps and at
# 15 20 30 50 70 100 150 200 300; every checkpoint merged (token-share weights) and scored on the reporting held-out sample (CE only), square 0
# alone scored at the log-spaced steps, all in ONE 8-GPU job (eval_batch_8gpu.sh); step checkpoints and merged weights deleted afterwards.
#   bash scripts/sparse_experts/olmoe3_squares/squares_steps_std.sh k8|k4 [std|mix20]      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k4|k8}"; MODEL="${2:-std}"; LR=4e-4; STEPS="${STEPS:-1 2 3 4 5 6 7 8 9 10 15 20 30 50 70 100 150 200 300}"; N=${STEPS##* }; SUB0_STEPS="15 20 30 50 70 100 150 200 300"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz
case $MODEL in
  std)   FULL=olmoe3_275m_10b; P4=olmoe3_squares_stdrand; P8=olmoe3_squares_stdrand8; TAG=stdrand; MENV="OLMOE3_EMO=0"; EXTRA=std;;   # standard routing; control arms at LR 8e-4
  mix20) FULL=olmoe3_275m_emo_mix20_10b; P4=olmoe3_squares_mix20rand4; P8=olmoe3_squares_mix20rand8; TAG=mix20rand; MENV="OLMOE3_EMO=1 OLMOE3_EMO_POOL_SELECT=mix:0.2"; EXTRA=mix20;;   # mixed-pool EMO; control arms at LR 4e-4 (exact prefixes)
  *) echo "std|mix20"; exit 1;;
esac
case $KV in k8) K=8; PARENT=$P8; SQN=olmoe3_squares_${TAG}8_lr4e-4_steps300; RP=olmoe3_275m_${TAG}8_lr4e-4_steps300_square; HR=runs_heldout300b_${TAG}8_lr4e-4_steps300; JOBS="0,1,2,3 4,5,6,7";;
            k4) K=4; PARENT=$P4; SQN=olmoe3_squares_${TAG}_lr4e-4_steps300;  RP=olmoe3_275m_${TAG}_lr4e-4_steps300_square;  HR=runs_heldout300b_${TAG}_lr4e-4_steps300;  JOBS="0,1,2,3";; *) echo "k4|k8"; exit 1;; esac
[ $MODEL = mix20 ] && [ $KV = k4 ] && { SQN=olmoe3_squares_mix20rand4_lr4e-4_steps300; RP=olmoe3_275m_mix20rand4_lr4e-4_steps300_square; HR=runs_heldout300b_mix20rand4_lr4e-4_steps300; }   # the mix20 4-square control is named ...rand4
SQ=$S/$SQN; LOG=$SQ/logs; mkdir -p $SQ $LOG $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [${MODEL}steps $KV] $*"; }
[ -f $SQ/groups.json ] || cp $S/$PARENT/groups.json $SQ/groups.json
[ -e $SQ/pack ] || ln -s ../$PARENT/pack $SQ/pack
# start slices: the control arm's init/ when it still exists (symlink), else re-sliced from the start model
[ -e $SQ/init ] || { [ -f $S/$PARENT/init/group0/.metadata ] && ln -s ../$PARENT/init $SQ/init; }
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] && continue
  { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=4 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; say "slice group$g: $(tail -n 1 $LOG/slice_group$g.log | cut -c1-100)"; } & done; wait
for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice $SQ/init/group$g missing"; exit 1; }; done
launch_job() { local name=$1 groups=$2 mark=$LOG/worker_$1.launched; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP $MENV OLMOE3_LR=$LR STEPS=$N CKPT_STEPS=${STEPS// /,} OLMOE3_WANDB_TAGS=$SQN,square,first_steps,$EXTRA bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh ${groups//,/ }" 2>&1 | tail -n 1)
  say "$name: $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
i=0; for groups in $JOBS; do launch_job "${MODEL}steps-$KV-j$i" "$groups"; i=$((i+1)); done
SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
merge_step() { local s=$1; [ -f $SQ/merged/step$s/merge_info.json ] && return 0; local subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$s"); done
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged/step$s --overwrite 2>&1 | tail -n 1 | cut -c1-120; say "merged step $s"; }
have_all() { local s=$1 g; for g in $(seq 0 $((K-1))); do [ -f $S/${RP}$g/step$s/train/rank0.pt ] || return 1; done; }
until have_all $N; do sleep 120; done; say "all $K squares have their step-$N checkpoints"
n=0; for s in $STEPS; do merge_step $s & n=$((n+1)); [ $((n % 3)) = 0 ] && wait; done; wait; say "all merges done"
CF=$LOG/evals_8gpu.txt; : > $CF
ev() { local name=$1 ckpt=$2; [ -f $S/olmoe3_routing/$HR/$name/none/meta.json ] || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $ckpt --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$name/none --restrict none --batch-size 8 --log-every 100 --ce-only" >> $CF; }
for s in $STEPS; do ev merged_step$s $W/$SQN/merged/step$s; done; for s in $SUB0_STEPS; do ev sub0_step$s $W/${RP}0/step$s; done
if [ -s $CF ] && [ ! -f $LOG/eval8gpu.launched ]; then u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "${MODEL}steps-$KV-evals" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_8gpu.txt 2>&1 | tail -n 1)
  say "evals ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/eval8gpu.launched; fi
for s in $STEPS; do until [ -f $S/olmoe3_routing/$HR/merged_step$s/none/meta.json ]; do sleep 120; done; done; for s in $SUB0_STEPS; do until [ -f $S/olmoe3_routing/$HR/sub0_step$s/none/meta.json ]; do sleep 120; done; done; say "all evaluations done"
for s in $STEPS; do for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g/step${s:?}"; done; rm -rf "$SQ/merged/step${s:?}/model_and_optim"; done
say "step checkpoints and merged weights deleted; merged: $(for s in $STEPS; do echo -n "s$s=$(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/merged_step$s/none/meta.json'))['mean_ce'],4))") "; done)"; say done
