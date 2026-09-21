#!/usr/bin/env bash
# Re-merge + re-partition test (user request 2026-09-20): take the standard-MoE random-control merge at 61.1B tokens (window-3 point
# 116,479 = the four random squares merged), merge it WITH Adam state, re-split its experts into four NEW random groups (seed 1,
# different from the training partition) and continue four squares on contiguous quarters of the remaining stream (steps
# 116,479 -> 247,956, 32,870 steps each), merging with equal weights at the same matched points as the control (166,478 = 87.3B,
# 216,477 = 113B, 247,956 = 130B). Hypothesis: re-merging and re-partitioning stops the squares from diverging.
#   bash scripts/sparse_experts/olmoe3_squares/squares_stdremerge.sh   (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_stdremerge; SQ=$S/$SQN; SRC=$S/olmoe3_squares_stdrand
FULL=olmoe3_275m_10b; RP=olmoe3_275m_stdremerge_square; HR=runs_heldout300b_stdremerge; SAMPLE=sample_8k_300b.npz
B0=116479; SQ_STEPS=32870; OFFS=(12500 25000 32870); PTS=(166478 216477 247956); SRC31=(72036 119720 167404 215088)  # the control squares' 31% checkpoints
LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in 1 2 3 4 5 6; do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; have $S/olmoe3_routing/$HR/$tag/none || launch "$SQN-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
# ---- 1. merge the 61.1B control squares WITH Adam state; new random expert groups; slice; finetune starts ----
subs=""; for g in 0 1 2 3; do subs="$subs,$S/olmoe3_275m_stdrand_square${g}_w3/step${SRC31[$g]}"; done; subs=${subs#,}
[ -f $SQ/merged_optim/match116479/merge_info.json ] || { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SRC/groups.json --full $S/$FULL/step19074 --subs "$subs" --weights 0.25,0.25,0.25,0.25 --out $SQ/merged_optim/match116479 --with-optim --overwrite 2>&1 | tail -1 | cut -c1-120; say "61B merge with optimizer state done"; }
[ -f $SQ/merged_optim/match116479/config.json ] || cp $S/$FULL/step19074/config.json $SQ/merged_optim/match116479/config.json
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k 4 --seed 1
python - <<'PY'
import json; a=json.load(open("sparse_experts/olmoe3_squares_stdrand/groups.json")); b=json.load(open("sparse_experts/olmoe3_squares_stdremerge/groups.json"))
ov=[len(set(a["groups"][l][g]) & set(b["groups"][l][g])) for l in map(str, range(2, 10)) for g in range(4)]
print(f"new partition vs training partition: same-index group overlap {min(ov)}-{max(ov)} of 128 experts (32 expected at random)")
PY
for g in 0 1 2 3; do start=$((B0 + g * SQ_STEPS))
  [ -f $SQ/init/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $SQ/merged_optim/match116479 --groups $SQ/groups.json --group $g --out $SQ/init/group$g/model_and_optim 2>&1 | tail -1; say "slice group$g done"; }
  [ -f $SQ/ft_start/group$g/step$start/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/init/group$g --train-from $S/olmoe3_275m_30b_1node/step57221 --start-step $start --out $SQ/ft_start/group$g 2>&1 | tail -1
  fixed=$(python -c "b=$start; print(','.join(str(b + o) for o in (${OFFS[0]}, ${OFFS[1]}, ${OFFS[2]})))")
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_stdrand_square_w3.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN RUN_PREFIX=$RP OLMOE3_EMO=0 OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square FT_START=$W/$SQN/ft_start/group$g FT_START_STEP=$start FT_STEPS=$SQ_STEPS OLMOE3_FIXED_STEPS=$fixed
done
# ---- 2. matched points: merge + evals ----
for i in 0 1 2; do s=${PTS[$i]}; subs=(); for g in 0 1 2 3; do subs+=("$S/${RP}$g/step$((B0 + g * SQ_STEPS + OFFS[$i]))"); done
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 600; done; done; say "point $s: square checkpoints present"
  [ -f $SQ/merged/match$s/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "${subs[*]}")" --weights 0.25,0.25,0.25,0.25 --out $SQ/merged/match$s --overwrite 2>&1 | tail -1 | cut -c1-120
  heldout merged_match$s $W/$SQN/merged/match$s; ppl $W/$SQN/merged/match$s merged/match$s.json merged-$s
  for g in 0 1 2 3; do heldout sub${g}_match$s $W/${RP}$g/step$((B0 + g * SQ_STEPS + OFFS[$i])); done
done
# ---- 3. collect ----
expected=""; for s in "${PTS[@]}"; do expected="$expected merged_match$s"; for g in 0 1 2 3; do expected="$expected sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$HR/$tag/none; do sleep 300; done; done
for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "re-merge/re-partition run done"
