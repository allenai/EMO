#!/usr/bin/env bash
# Branch-point experiment (user request 2026-11-15): where between the random init and 10B does splitting start to cost? The joint
# standard-MoE 512e run olmoe3_275m_scratch_joint_1b is checkpointed at ten log-spaced points of its first 1B tokens; from each branch
# point <B> (a joint step) the model is cut into <K> random squares (the from-scratch arms' expert groups) that train for the NEXT 100M
# tokens of the stream in total (joint steps B .. B+191; the documents of those steps split at random, the from-scratch arms'
# assignment), i.e. about 191/K steps each. Optimizer moments carried; the squares continue the joint run's LR ramp at the same tokens
# (peak 8e-4 reached after (2000-B)/K square steps, starting from the joint LR at B). Every 24/K square steps (= every 24 joint batches,
# 12.6M tokens: 8 points per branch) and at the last step the squares are checkpointed, merged (token-share weights) and the merge and
# every square alone are scored on the reporting held-out sample; `joint` scores the joint run at every branch point and 48 / 96 / 144 /
# 191 steps later. At most 3 branches are in flight at a time (their checkpoints take ~0.5 TB each). Everything a branch wrote (packs, slices, square runs, merged weights) is deleted after scoring; the
# joint run's branch-point checkpoints are kept, its B+191 checkpoints deleted.
#   bash scripts/sparse_experts/olmoe3_squares/squares_branch.sh <B> <K> | joint | all      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
BS="10 25 50 100 200 400 800 1300 1907"; SPAN=191   # branch point 0 = the from-scratch arms (squares_scratch*.sh), not repeated (user 2026-11-15)
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; JOINT=olmoe3_275m_scratch_joint_1b; ST=$S/olmoe3_routing/stream_0b_10b
HR=runs_heldout300b_branch; H=$S/olmoe3_routing/$HR; ROOTD=$S/olmoe3_squares_branch; mkdir -p $H $ROOTD/logs
evline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none --restrict none --batch-size 8 --log-every 100 --ce-only --max-first-ce 100"; }
done_ev() { [ -s $H/$1/none/meta.json ]; }
ce() { python -c "import json; print(round(json.load(open('$H/$1/none/meta.json'))['mean_ce'],4))" 2>/dev/null; }
launch_evals() { local name=$1 cf=$2 mark=$3; [ -s $cf ] || return 0; [ -f $mark ] && return 0; local u
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh ${cf/$S/$W} 2>&1 | tail -n 1)
  say "evals $name ($(wc -l < $cf) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; }
# at most 3 branches in flight, and only with > 1.2 TB free (slice -> train -> merge -> score -> delete); a slot is re-entered by its own branch after a restart and is stale after 12 h
slot_get() { local i; for i in 0 1 2; do [ "$(cat $ROOTD/logs/slot$i/owner 2>/dev/null)" = "$TAG" ] && { SLOT=$i; return; }; done
  until [ "$(df --output=avail -BG /root | tail -1 | tr -dc 0-9)" -gt 1200 ]; do sleep 300; done   # a branch needs ~0.5 TB until it is scored
  while true; do for i in 0 1 2; do mkdir $ROOTD/logs/slot$i 2>/dev/null && { echo $TAG > $ROOTD/logs/slot$i/owner; SLOT=$i; return; }; [ $(( $(date +%s) - $(stat -c %Y $ROOTD/logs/slot$i 2>/dev/null || date +%s) )) -gt 43200 ] && rm -rf $ROOTD/logs/slot$i; done; sleep 30; done; }
slot_put() { rm -rf "$ROOTD/logs/slot${SLOT:?}"; }
OFFS="48 96 144 191"
MODE="${1:?<B> <K> | joint | all}"
if [ $MODE = all ]; then   # start every driver that is not running (after a session restart: rerun this)
  for a in joint $(for b in $BS; do echo "$b:4 $b:8"; done); do set -- ${a/:/ }; pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/squares_branch.sh $*\$" > /dev/null && continue
    setsid nohup bash scripts/sparse_experts/olmoe3_squares/squares_branch.sh "$@" >> $ROOTD/logs/driver_${a/:/_k}.log 2>&1 < /dev/null & done; exit 0; fi
if [ $MODE = joint ]; then say() { echo "$(date -u +%m-%d\ %H:%M) [branch joint] $*"; }; LOG=$ROOTD/logs/joint; mkdir -p $LOG
  JS=$(for b in $BS; do echo $b; for o in $OFFS; do echo $((b + o)); done; done | sort -n | uniq | tr '\n' ' ')
  # scores the checkpoints that exist now and prunes them (rerun after the trunk's last steps land: 2026-11-15 the disk filled at step 1955)
  JS=$(for s in $JS; do [ -f $S/$JOINT/step$s/train/rank0.pt ] || done_ev joint_step$s && echo -n "$s "; done); say "scoring joint steps: $JS"
  CF=$LOG/evals.txt; : > $CF; for s in $JS; do done_ev joint_step$s || evline joint_step$s $W/$JOINT/step$s >> $CF; done; rm -f $LOG/evals.launched; launch_evals branch-joint $CF $LOG/evals.launched
  for s in $JS; do until done_ev joint_step$s; do sleep 120; done; done; say "joint: $(for s in $JS; do echo -n "s$s=$(ce joint_step$s) "; done)"
  # only the branch points themselves are kept
  for s in $JS; do case " $BS " in *" $s "*) ;; *) rm -rf "$S/${JOINT:?}/step${s:?}";; esac; done; say done; exit 0; fi
B=$MODE; K="${2:?K}"; TAG=b${B}_k$K; SQN=olmoe3_squares_branch/$TAG; SQ=$S/$SQN; LOG=$ROOTD/logs/$TAG; RP=olmoe3_275m_branch_${TAG}_square; SRC=$S/olmoe3_squares_scratch$K; mkdir -p $LOG
say() { echo "$(date -u +%m-%d\ %H:%M) [branch $TAG] $*"; }
allsc() { [ -s $LOG/steps.txt ] || return 1; local s g; for s in $(cat $LOG/steps.txt); do done_ev ${TAG}_merged_s$s || return 1; for g in $(seq 0 $((K-1))); do done_ev ${TAG}_sub${g}_s$s || return 1; done; done; }
if ! allsc; then mkdir -p $SQ
  until [ -f $S/$JOINT/step$B/train/rank0.pt ]; do sleep 120; done
  slot_get
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
  CK=$(python -c "st=24//$K; l=sorted(set(range(st, $N + 1, st)) | {$N}); l=[x for x in l if x == $N or $N - x >= st // 2 + 1]; print(' '.join(map(str, l)))"); echo "$CK" > $LOG/steps.txt
  WARM=$(python -c "print(max(1, -(-(2000 - $B) // $K)))"); MINLR=$(python -c "print(8e-4 * $B / 2000)")
  for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] && continue
    { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=4 python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$JOINT/step$B --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1; } & done; wait
  for g in $(seq 0 $((K-1))); do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice group$g missing"; exit 1; }; done
  if [ ! -f $LOG/worker.launched ]; then
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "branch-$TAG-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=0 OLMOE3_LR=8e-4 OLMOE3_WARMUP=$WARM OLMOE3_WARMUP_MIN_LR=$MINLR STEPS=$N CKPT_STEPS=${CK// /,} OLMOE3_WANDB_TAGS=olmoe3_squares_branch,square,$TAG bash scripts/sparse_experts/olmoe3_squares/steps_worker.sh $(seq -s' ' 0 $((K-1)))" 2>&1 | tail -n 1)
    say "squares ($N steps each, checkpoints at $CK, warmup $WARM from LR $MINLR): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $LOG/worker.launched; fi
  # as each square finishes, its checkpoints are cut down to the weights (a third of the size; strip_optim.py) -- the disk filled twice on 2026-11-15
  for g in $(seq 0 $((K-1))); do until [ -f $S/${RP}$g/step$N/train/rank0.pt ]; do sleep 60; done; sleep 90
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/strip_optim.py $(for s in $CK; do echo $S/${RP}$g/step$s; done) >> $LOG/strip.log 2>&1; done
  for g in $(seq 0 $((K-1))); do for s in $CK; do [ -f $S/${RP}$g/step$s/.main_only ] || { say "ABORT: strip of square $g step $s failed"; exit 1; }; done; done; say "all $K squares done and stripped"
  SH=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
  n=0; for s in $CK; do [ -f $SQ/merged/s$s/merge_info.json ] || { subs=(); for g in $(seq 0 $((K-1))); do subs+=("$S/${RP}$g/step$s"); done
      PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$JOINT/step$B --subs "$(IFS=,; echo "${subs[*]}")" --weights "$SH" --out $SQ/merged/s$s --overwrite > $LOG/merge_s$s.log 2>&1 & n=$((n+1)); [ $((n % 3)) = 0 ] && wait; }; done; wait
  for s in $CK; do [ -f $SQ/merged/s$s/merge_info.json ] || { say "ABORT: merge at step $s failed"; exit 1; }; done
  CF=$LOG/evals.txt; : > $CF; for s in $CK; do done_ev ${TAG}_merged_s$s || evline ${TAG}_merged_s$s $W/$SQN/merged/s$s >> $CF
    for g in $(seq 0 $((K-1))); do done_ev ${TAG}_sub${g}_s$s || evline ${TAG}_sub${g}_s$s $W/${RP}$g/step$s >> $CF; done; done; launch_evals branch-$TAG $CF $LOG/evals.launched
  # an evaluation job that exits with passes still missing (2026-11-15: results lost to a full disk although the job reported success) is resubmitted, up to 3 times
  tries=0; until allsc; do sleep 180; e=$(grep -o '01[A-Z0-9]\{20,\}' $LOG/evals.launched 2>/dev/null | head -1)
    if [ -n "$e" ] && beaker experiment get $e --format json 2>/dev/null | grep -q '"exited"' && ! allsc; then sleep 60; allsc && break; tries=$((tries+1)); [ $tries -gt 3 ] && { say "ABORT: evaluations still missing after 3 resubmissions"; exit 1; }
      : > $CF; for s in $CK; do done_ev ${TAG}_merged_s$s || evline ${TAG}_merged_s$s $W/$SQN/merged/s$s >> $CF; for g in $(seq 0 $((K-1))); do done_ev ${TAG}_sub${g}_s$s || evline ${TAG}_sub${g}_s$s $W/${RP}$g/step$s >> $CF; done; done
      rm -f $LOG/evals.launched; say "resubmitting $(wc -l < $CF) missing evaluations (try $tries)"; launch_evals branch-$TAG $CF $LOG/evals.launched; fi; done
fi
for g in $(seq 0 $((K-1))); do rm -rf "$S/${RP:?}$g"; done; rm -rf "${SQ:?}/merged" "${SQ:?}/init" "${SQ:?}/pack" "${SQ:?}/assign"
for i in 0 1 2; do [ "$(cat $ROOTD/logs/slot$i/owner 2>/dev/null)" = "$TAG" ] && rm -rf "$ROOTD/logs/slot$i"; done
say "merged: $(for s in $(cat $LOG/steps.txt); do echo -n "s$s=$(ce ${TAG}_merged_s$s) "; done)"; say done
