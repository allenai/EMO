#!/usr/bin/env bash
# Block A's SPECTRAL expert groups with a RANDOM EQUAL document split (user request 2026-09-29): the EMO 512e model split into the same
# four block-group sub-models as block A (groups.json and sliced start checkpoints reused unchanged), but every document of the 10B -> 20B
# window goes to a uniformly random square (the random equal packs of the Q5 controls, olmoe3_squares_stdrand/pack: ~2.5B tokens each)
# instead of the square its routing picked. Everything else as block A: EMO loss, LR 8e-4, checkpoints at the five matched fractions,
# merged there (token-share weights) and evaluated on the 20B-window held-out sample and the v3-small ppl sets. No finetuning, no
# piecewise passes. Idempotent; detach.
#   bash scripts/sparse_experts/olmoe3_squares/squares_rnddoc.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_rnddoc; SQ=$S/$SQN; RP=olmoe3_275m_emornddoc_square
FULL=olmoe3_275m_emo_10b; HR=runs_heldout20b_rnddoc; LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [rnddoc] $*"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
# ---- 0. block A's groups and start slices; the random equal document packs ----
[ -f $SQ/groups.json ] || cp $S/olmoe3_squares/groups.json $SQ/groups.json
[ -e $SQ/init ] || ln -s ../olmoe3_squares/init $SQ/init
[ -e $SQ/pack ] || ln -s ../olmoe3_squares_stdrand/pack $SQ/pack
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
# ---- 1. the four squares (1 node each, allocated) ----
for g in 0 1 2 3; do
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square,random_documents
done
declare -a STEPS
for g in 0 1 2 3; do STEPS[$g]=$(python -c "
import json; t=json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g]; s=t//524288
print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')"); done
say "square steps: g0 [${STEPS[0]}] g1 [${STEPS[1]}] g2 [${STEPS[2]}] g3 [${STEPS[3]}]; token shares $SHARES"
# ---- 2. matched points: merge + evaluations ----
for i in 0 1 2 3 4; do
  [ -f $LOG/merge_point$i.log ] && grep -q "ppl:" $LOG/merge_point$i.log || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES SKIP_ORACLE=1 \
    STEPS_G0="${STEPS[0]}" STEPS_G1="${STEPS[1]}" STEPS_G2="${STEPS[2]}" STEPS_G3="${STEPS[3]}" bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
done
wait
for d in $S/olmoe3_routing/$HR/*/none; do until [ -f $d/rank0/DONE ] || [ -f $d/meta.json ]; do sleep 120; done; [ -f $d/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
