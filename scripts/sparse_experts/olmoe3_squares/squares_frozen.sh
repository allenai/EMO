#!/usr/bin/env bash
# Frozen-router squares for Q3 block A (user request 2026-09-24): the EMO 512e model's four block-group sub-models, with exactly the
# same partition (groups.json), document assignment (pack, routing-based: each document to the group receiving most of its layer 2-9
# selections) and sliced start checkpoints (init) as block A, trained on their own documents of the 10B -> 20B window with every
# routed-expert router at LR 0 (OLMOE3_FREEZE_ROUTER=1), merged at the five matched points (experts scattered back, shared parameters
# token-share averaged; identical tensors copied exactly) and evaluated like block A (held-out 20B-window CE, v3-small ppl). After every
# merge, check_router_frozen.py verifies that the merged routers are bit-identical to the start model's. Idempotent; detach.
#   bash scripts/sparse_experts/olmoe3_squares/squares_frozen.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SRC=$S/olmoe3_squares; SQN=olmoe3_squares_frz; SQ=$S/$SQN; RP=olmoe3_275m_emofrz_square
FULL=olmoe3_275m_emo_10b; HR=runs_heldout20b_frz; LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [frz] $*"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
# ---- 0. share block A's partition, packs and sliced start checkpoints (same storage on weka and in this session) ----
[ -f $SQ/groups.json ] || cp $SRC/groups.json $SQ/groups.json
[ -e $SQ/pack ] || ln -s ../olmoe3_squares/pack $SQ/pack
[ -e $SQ/init ] || ln -s ../olmoe3_squares/init $SQ/init
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
# ---- 1. the four squares (1 node each, allocated; routers frozen) ----
for g in 0 1 2 3; do
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square,router_frozen OLMOE3_FREEZE_ROUTER=1
done
declare -a STEPS
for g in 0 1 2 3; do STEPS[$g]=$(python -c "
import json; t=json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g]; s=t//524288
print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')"); done
say "square steps: g0 [${STEPS[0]}] g1 [${STEPS[1]}] g2 [${STEPS[2]}] g3 [${STEPS[3]}]; token shares $SHARES"
# ---- 2. matched points: merge + evaluations (merge_point.sh waits for the checkpoints) + router sanity check ----
for i in 0 1 2 3 4; do
  [ -f $LOG/merge_point$i.log ] && grep -q "ppl:" $LOG/merge_point$i.log || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES SKIP_ORACLE=1 \
    STEPS_G0="${STEPS[0]}" STEPS_G1="${STEPS[1]}" STEPS_G2="${STEPS[2]}" STEPS_G3="${STEPS[3]}" bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
done
BASE=(20000 25000 30000 35000 38148)
for i in 0 1 2 3 4; do s=${BASE[$i]}
  until [ -f $SQ/merged/match$s/merge_info.json ]; do sleep 300; done
  [ -f $LOG/check_frozen_match$s.json ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/check_router_frozen.py --start $S/$FULL/step19074 --merged $SQ/merged/match$s --json $LOG/check_frozen_match$s.json 2>&1 | tail -1 | sed "s/^/match$s: /"
done
wait
for d in $S/olmoe3_routing/$HR/*/none; do until [ -f $d/rank0/DONE ] || [ -f $d/meta.json ]; do sleep 120; done; [ -f $d/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
