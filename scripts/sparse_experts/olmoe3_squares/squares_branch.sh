#!/usr/bin/env bash
# Branch-point experiment (user request 2026-11-15): where between the random init and 10B does splitting start to cost? The joint
# standard-MoE 512e run olmoe3_275m_scratch_joint_1b is checkpointed at ten log-spaced points of its first 1B tokens; from each branch
# point <B> (a joint step) the model is cut into <K> random squares (the from-scratch arms' expert groups) that train for the NEXT 100M
# tokens of the stream in total (joint steps B .. B+191; the documents of those steps split at random, the from-scratch arms'
# assignment), i.e. about 191/K steps each. Optimizer moments carried; the squares continue the joint run's LR ramp at the same tokens
# (peak 8e-4 reached after (2000-B)/K square steps, starting from the joint LR at B). At the end the squares are merged (token-share
# weights) and the merge and every square alone are scored on the reporting held-out sample; `joint` scores the joint run at every
# branch point and 191 steps later. Everything a branch wrote (packs, slices, square runs, merged weights) is deleted after scoring; the
# joint run's branch-point checkpoints are kept, its B+191 checkpoints deleted.
#   bash scripts/sparse_experts/olmoe3_squares/squares_branch.sh <B> <K> | joint | all      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
BS="0 10 25 50 100 200 400 800 1300 1907"; SPAN=191
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; JOINT=olmoe3_275m_scratch_joint_1b; ST=$S/olmoe3_routing/stream_0b_10b
HR=runs_heldout300b_branch; H=$S/olmoe3_routing/$HR; ROOTD=$S/olmoe3_squares_branch; mkdir -p $H $ROOTD/logs
evline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none --restrict none --batch-size 8 --log-every 100 --ce-only --max-first-ce 100"; }
done_ev() { [ -f $H/$1/none/meta.json ]; }
ce() { python -c "import json; print(round(json.load(open('$H/$1/none/meta.json'))['mean_ce'],4))" 2>/dev/null; }
launch_evals() { local name=$1 cf=$2 mark=$3; [ -s $cf ] || return 0; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh ${cf/$S/$W} 2>&1 | tail -n 1)
  say "evals $name ($(wc -l < $cf) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
# at most 5 branches slice or merge at a time (each loads a 31 GB checkpoint per process); a slot older than 2 h is stale
slot_get() { local i; while true; do for i in 0 1 2 3 4; do mkdir $ROOTD/logs/slot$i 2>/dev/null && { SLOT=$i; return; }; [ $(( $(date +%s) - $(stat -c %Y $ROOTD/logs/slot$i 2>/dev/null || date +%s) )) -gt 7200 ] && rmdir $ROOTD/logs/slot$i 2>/dev/null; done; sleep 20; done; }
slot_put() { rmdir $ROOTD/logs/slot$SLOT 2>/dev/null; }
MODE="${1:?<B> <K> | joint | all}"
if [ $MODE = all ]; then   # start every driver that is not running (after a session restart: rerun this)
  for a in joint $(for b in $BS; do echo "$b:4 $b:8"; done); do set -- ${a/:/ }; pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/squares_branch.sh $*\$" > /dev/null && continue
    setsid nohup bash scripts/sparse_experts/olmoe3_squares/squares_branch.sh "$@" >> $ROOTD/logs/driver_${a/:/_k}.log 2>&1 < /dev/null & done; exit 0; fi
if [ $MODE = joint ]; then say() { echo "$(date -u +%m-%d\ %H:%M) [branch joint] $*"; }; LOG=$ROOTD/logs/joint; mkdir -p $LOG
  JS=$(for b in $BS; do echo $b $((b + SPAN)); done | tr ' ' '\n' | sort -n | uniq | tr '\n' ' ')
  until [ -f $S/$JOINT/step$((1907 + SPAN))/train/rank0.pt ]; do sleep 120; done; say "joint run complete"
  CF=$LOG/evals.txt; : > $CF; for s in $JS; do done_ev joint_step$s || evline joint_step$s $W/$JOINT/step$s >> $CF; done; launch_evals branch-joint $CF $LOG/evals.launched
  for s in $JS; do until done_ev joint_step$s; do sleep 120; done; done; say "joint: $(for s in $JS; do echo -n "s$s=$(ce joint_step$s) "; done)"
  # the B+191 checkpoints go once every branch has been scored (the branch points themselves are kept)
  for b in $BS; do for k in 4 8; do until done_ev b${b}_k${k}_merged; do sleep 300; done; done; done
  for b in $BS; do s=$((b + SPAN)); case " $BS " in *" $s "*) ;; *) rm -rf "$S/${JOINT:?}/step${s:?}";; esac; done; say done; exit 0; fi
B=$MODE; K="${2:?K}"; TAG=b${B}_k$K; SQN=olmoe3_squares_branch/$TAG; SQ=$S/$SQN; LOG=$ROOTD/logs/$TAG; RP=olmoe3_275m_branch_${TAG}_square; SRC=$S/olmoe3_squares_scratch$K; mkdir -p $LOG
say() { echo "$(date -u +%m-%d\ %H:%M) [branch $TAG] $*"; }
allsc() { done_ev ${TAG}_merged || return 1; local g; for g in $(seq 0 $((K-1))); do done_ev ${TAG}_sub$g || return 1; done; }
if ! allsc; then mkdir -p $SQ
  until [ -f $S/$JOINT/step$B/train/rank0.pt ]; do sleep 120; done
  # ---- groups, the documents of joint steps B .. B+191 (the from-scratch arms' random assignment), packs, start slices ----
  [ -f $SQ/groups.json ] || cp $SRC/groups.json $SQ/groups.json
  [ -f $SQ/assign/records_0.npz ] || python - <<PY
import numpy as np, glob, os
lo, hi = $B * 64, ($B + $SPAN) * 64; R = {}
for f in sorted(glob.glob("$SRC/assign/records_*.npz")):
    z = np.load(f); m = (z["inst"] >= lo) & (z["inst"] < hi)
    if m.any():
        for k in z.files: R.setdefault(k, []).append(z[k][m])
os.makedirs("$SQ/assign", exist_ok=True); np.savez("$SQ/assign/records_0.npz", **{k: np.concatenate(v) for k, v in R.items()})
PY
  [ -f $SQ/pack/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign --stream $ST --out $SQ/pack --groups $SQ/groups.json > $LOG/pack.log 2>&1; say "pack: $(tail -n 1 $LOG/pack.log | cut -c1-140)"; }
  [ -f $SQ/pack/stats.json ] || { say "ABORT: pack failed"; exit 1; }
  N=$(python -c "import json; print(min(t // 524288 for t in json.load(open('$SQ/pack/stats.json'))['tokens_per_group']))")
  WARM=$(python -c "print(max(1, -(-(2000 - $B) // $K)))"); MINLR=$(python -c "print(8e-4 * $B / 2000)")
  slot_get; for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] && continue
    { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=4 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$JOINT/step$B --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; } & done; wait; slot_put
  for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice group$g missing"; exit 1; }; done
  if [ ! -f $LOG/worker.launched ]; then
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "branch-$TAG-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=0 OLMOE3_LR=8e-4 OLMOE3_WARMUP=$WARM OLMOE3_WARMUP_MIN_LR=$MINLR STEPS=$N CKPT_STEPS=$N OLMOE3_WANDB_TAGS=olmoe3_squares_branch,square,$TAG bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh $(seq -s' ' 0 $((K-1)))" 2>&1 | tail -n 1)
    say "squares ($N steps each, warmup $WARM from LR $MINLR): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/worker.launched; fi
  for g in $(seq 0 $((K-1))); do until [ -f $S/${RP}$g/step$N/train/rank0.pt ]; do sleep 120; done; done; say "all $K squares done"
  SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
  if [ ! -f $SQ/merged/merge_info.json ]; then slot_get; subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$N"); done
    PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$JOINT/step$B --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged --overwrite 2>&1 | tail -n 1 | cut -c1-100; slot_put; fi
  [ -f $SQ/merged/merge_info.json ] || { say "ABORT: merge failed"; exit 1; }
  CF=$LOG/evals.txt; : > $CF; done_ev ${TAG}_merged || evline ${TAG}_merged $W/$SQN/merged >> $CF
  for g in $(seq 0 $((K-1))); do done_ev ${TAG}_sub$g || evline ${TAG}_sub$g $W/${RP}$g/step$N >> $CF; done; launch_evals branch-$TAG $CF $LOG/evals.launched
  until allsc; do sleep 120; done
fi
for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g"; done; rm -rf "${SQ:?}/merged/model_and_optim" "${SQ:?}/init" "${SQ:?}/pack" "${SQ:?}/assign"
say "merged $(ce ${TAG}_merged); squares alone $(for g in $(seq 0 $((K-1))); do echo -n "$(ce ${TAG}_sub$g) "; done)"; say done
