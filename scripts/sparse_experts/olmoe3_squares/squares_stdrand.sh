#!/usr/bin/env bash
# Random-partition control for the standard-MoE 512e squares (report Q3 block B; user request 2026-09-18): four random expert
# groups of 128 per layer (layer 1 whole) and the window-1 documents (10B -> 20B, stream steps 19,074-38,147) split uniformly at
# random into four groups; one square per group from the sliced 10B start model, same training recipe as block B; merged with
# equal weights at the five matched points; merged model and every square evaluated on the 300B-token held-out sample.
#   bash scripts/sparse_experts/olmoe3_squares/squares_stdrand.sh      (idempotent; detach it; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_stdrand; SQ=$S/$SQN; SRC=$S/olmoe3_squares_std
FULL=olmoe3_275m_10b; RP=olmoe3_275m_stdrand_square; HR=runs_heldout300b_stdrand; SAMPLE=sample_8k_300b.npz; ST=$S/olmoe3_routing/stream_10b_20b
STEPS=(20000 25000 30000 35000 38148); LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $S/olmoe3_routing/$HR $SP
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; have $S/olmoe3_routing/$HR/$tag/none || launch "$SQN-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
# ---- 0. random expert groups, random document groups, packs, sliced start checkpoints ----
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k 4 --seed 0
[ -f $SQ/assign/records_7.npz ] || python scripts/sparse_experts/olmoe3_squares/assign_random.py --src $SRC/assign --out $SQ/assign --k 4 --seed 0
[ -f $SQ/pack/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $SQ/assign --stream $ST --out $SQ/pack --groups $SQ/groups.json 2>&1 | tail -2; say "pack done"; }
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
for g in 0 1 2 3; do [ -f $SQ/init/group$g/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g 2>&1 | tail -1; say "slice group$g done"; }; done
declare -a SQS
for g in 0 1 2 3; do SQS[$g]=$(python -c "import json; t=json.load(open('$SQ/pack/stats.json'))['tokens_per_group'][$g]; s=t//524288; print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')"); done
say "square steps: g0 [${SQS[0]}] g1 [${SQS[1]}] g2 [${SQS[2]}] g3 [${SQS[3]}]; token shares $SHARES"
# ---- 1. squares (one node each) ----
for g in 0 1 2 3; do
  if [ ! -f $LOG/square${g}_launched ]; then u=""
    for attempt in 1 2 3 4 5 6; do SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_std_square.sh launch > $LOG/launch_square$g.log 2>&1
      u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/launch_square$g.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
    say "square$g: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/square${g}_launched; fi
done
# ---- 2. matched points: merge (waits for the checkpoints) + merged evals on the 300B sample ----
for i in 0 1 2 3 4; do
  [ -f $SQ/merged/match${STEPS[$i]}/merge_info.json ] && [ -f $SQ/ppl_validation/merged/match${STEPS[$i]}.json ] || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES SKIP_ORACLE=1 SAMPLE=$SAMPLE \
    STEPS_G0="${SQS[0]}" STEPS_G1="${SQS[1]}" STEPS_G2="${SQS[2]}" STEPS_G3="${SQS[3]}" bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
done
# ---- 3. every square on the whole sample at every point ----
for i in 0 1 2 3 4; do s=${STEPS[$i]}
  for g in 0 1 2 3; do st=(${SQS[$g]}); ck=$S/${RP}$g/step${st[$i]}; until [ -f $ck/train/rank0.pt ]; do sleep 300; done; heldout sub${g}_match$s $W/${RP}$g/step${st[$i]}; done
done
wait
# ---- 4. collect ----
expected=""; for s in "${STEPS[@]}"; do expected="$expected merged_match$s"; for g in 0 1 2 3; do expected="$expected sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$HR/$tag/none; do sleep 120; done; done
for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
python - <<'PY'
import json; from pathlib import Path
H=Path("sparse_experts/olmoe3_routing/runs_heldout300b_stdrand"); B=Path("sparse_experts/olmoe3_routing/runs_heldout300b_std")
ce=lambda d: round(json.load(open(d/"none/meta.json"))["mean_ce"],4) if (d/"none/meta.json").exists() else None
for s in (20000,25000,30000,35000,38148):
    print(s, "baseline", ce(B/f"baseline_step{s}"), "merged(random)", ce(H/f"merged_match{s}"), "squares", [ce(H/f"sub{g}_match{s}") for g in range(4)], "merged(routing)", ce(B/f"merged_match{s}"))
PY
say "random-partition control done"
