#!/usr/bin/env bash
# Random equal-size expert groups with block A's ROUTING-BASED document assignment (user request 2026-09-27): the EMO 512e model split
# into four sub-models whose expert groups are random and equal (128 experts per layer 2-9, layer 1 whole; partition_random seed 7)
# instead of the spectral block-groups, while every document keeps the square it was assigned to in block A (the group receiving most
# of its layer 2-9 routing: pack/). Everything else as block A: sliced start checkpoints (from the 10B model along the new groups),
# EMO loss, LR 8e-4, checkpoints at the five matched fractions, merged there (token-share weights) and evaluated on the 20B-window
# held-out sample and the v3-small ppl sets. No finetuning, no piecewise passes. Idempotent; detach.
#   bash scripts/sparse_experts/olmoe3_squares/squares_rndexp.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SRC=$S/olmoe3_squares; SQN=olmoe3_squares_rndexp; SQ=$S/$SQN; RP=olmoe3_275m_emorndexp_square
FULL=olmoe3_275m_emo_10b; HR=runs_heldout20b_rndexp; LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [rndexp] $*"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
# ---- 0. random equal groups; block A's document packs; start slices along the new groups ----
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k 4 --seed 7 > /dev/null
[ -e $SQ/pack ] || ln -s ../olmoe3_squares/pack $SQ/pack
python - <<'PY'
import json; a=json.load(open("sparse_experts/olmoe3_squares/groups.json")); b=json.load(open("sparse_experts/olmoe3_squares_rndexp/groups.json"))
ov=[len(set(a["groups"][l][g]) & set(b["groups"][l][g])) for l in map(str, range(2, 10)) for g in range(4)]
print(f"random groups vs block A's spectral groups: same-index overlap {min(ov)}-{max(ov)} experts (A's groups have 97-162; random ones 128 each)")
PY
for g in 0 1 2 3; do [ -f $SQ/init/group$g/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g 2>&1 | tail -1 | cut -c1-120; say "slice group$g done"; }; done
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
# ---- 1. the four squares (1 node each, allocated) ----
for g in 0 1 2 3; do
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square,random_expert_groups
done
declare -a STEPS
for g in 0 1 2 3; do STEPS[$g]=$(python -c "
import json; t=json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g]; s=t//524288
print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')"); done
say "square steps: g0 [${STEPS[0]}] g1 [${STEPS[1]}] g2 [${STEPS[2]}] g3 [${STEPS[3]}]; token shares $SHARES"
# ---- 2. matched points: merge + evaluations (merge_point.sh waits for the checkpoints) ----
for i in 0 1 2 3 4; do
  [ -f $LOG/merge_point$i.log ] && grep -q "ppl:" $LOG/merge_point$i.log || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES SKIP_ORACLE=1 \
    STEPS_G0="${STEPS[0]}" STEPS_G1="${STEPS[1]}" STEPS_G2="${STEPS[2]}" STEPS_G3="${STEPS[3]}" bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
done
wait
for d in $S/olmoe3_routing/$HR/*/none; do until [ -f $d/rank0/DONE ] || [ -f $d/meta.json ]; do sleep 120; done; [ -f $d/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
