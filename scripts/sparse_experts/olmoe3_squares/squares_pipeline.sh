#!/usr/bin/env bash
# End-to-end olmoe3_squares (k = 4) pipeline for one more start model, idempotent (every stage skips what exists):
#   stage 0 partition -> baseline continuation + start-model evals -> stage 1 assignment (8 x 1-GPU) + pack ->
#   init slices -> 4 square trainings -> merge/baseline evaluations at the 5 matched points -> square passes +
#   piecewise -> post-merge finetune stage 1 (+0.5B, merged and baseline) -> stage 2 (+1B, ft2.sh).
#   bash scripts/sparse_experts/olmoe3_squares/squares_pipeline.sh pool64or512|learnedd
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
TAG=$1
case $TAG in
  pool64or512) FULL=olmoe3_275m_emo_pool64or512_10b; COND=runs/emo_pool64or512/none; BASE_SCRIPT=olmoe3_275m_pool64or512_20b_baseline.sh
       EXTRA_ENV="OLMOE3_EMO_POOL_DIST=choice:64,512";;
  learnedd) FULL=olmoe3_275m_emo_learnedd_10b; COND=runs/emo_learnedd/none; BASE_SCRIPT=olmoe3_275m_learnedd_20b_baseline.sh
       EXTRA_ENV="OLMOE3_EMO_LEARNED_D=1 OLMOE3_LD_SIGNAL=coverage OLMOE3_LD_LAMBDA=1.0 OLMOE3_LD_LAMBDA_COV=1.0 OLMOE3_LD_TEMP=2.0 OLMOE3_LD_WARMUP=0 OLMOE3_LD_LR_MULT=10";;
  *) echo "pool64or512|learnedd"; exit 1;;
esac
SQN=olmoe3_squares_$TAG; RP=olmoe3_275m_${TAG}_square; HR=runs_heldout20b_$TAG; START=${TAG}_step19074; BRUN=olmoe3_275m_${TAG}_20b_1node
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
LOG=$S/$SQN/logs; mkdir -p $S/$SQN $LOG $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1; local log=$2; shift 2; [ -f "$marker" ] && return 0; local u=""
  for attempt in 1 2 3 4 5 6; do env "$@" OLMOE3_FOLLOW=0 bash -c 'bash "$0" launch' "$TRAIN_SCRIPT" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }

# ---- stage 0: partition ----
if [ ! -f $S/$SQN/groups.json ]; then
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/partition.py $S/olmoe3_routing/$COND --out $S/$SQN/groups.json --k 4 2>&1 | tail -2
  say "partition done"
fi
# ---- baseline continuation (1 node) + start-model evals ----
TRAIN_SCRIPT=scripts/sparse_experts/model_scripts/$BASE_SCRIPT launch_train $LOG/baseline_launched $LOG/launch_baseline.log $EXTRA_ENV
[ -f $S/olmoe3_routing/$HR/$START/none/meta.json ] || [ -f $S/olmoe3_routing/$HR/$START/none/rank0/DONE ] || \
  launch "$SQN-eval-start-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$FULL/step19074" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/$START/none/rank0" --restrict none --batch-size 8 --log-every 100
[ -f claude_outputs/debug_validation/ppl_validation/$FULL/step19074.json ] || \
  launch "$SQN-eval-start-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$FULL/step19074" --out-dir "/weka/oe-training-default/ryanwang/EMO/claude_outputs/debug_validation/ppl_validation" --batch-size 8
# ---- stage 1: assignment (8 ranks) + pack ----
for r in 0 1 2 3 4 5 6 7; do
  [ -f $S/$SQN/assign/records_$r.npz ] || [ -f $LOG/assign_r$r.launched ] || { launch "$SQN-assign-r$r" python scripts/sparse_experts/olmoe3_squares/assign_docs.py --checkpoint "$W/$FULL/step19074" --stream "$W/olmoe3_routing/stream_10b_20b" --groups "$W/$SQN/groups.json" --out "$W/$SQN/assign" --rank $r --world 8 --batch-size 8; touch $LOG/assign_r$r.launched; }
done
for r in 0 1 2 3 4 5 6 7; do until [ -f $S/$SQN/assign/records_$r.npz ]; do sleep 120; done; done; say "assignment records complete"
[ -f $S/$SQN/pack/stats.json ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/pack_groups.py --records $S/$SQN/assign --stream $S/olmoe3_routing/stream_10b_20b --out $S/$SQN/pack --groups $S/$SQN/groups.json 2>&1 | tail -2; say "pack done"; }
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$S/$SQN/pack/stats.json'))['token_share']))")
# ---- init slices ----
for g in 0 1 2 3; do
  [ -f $S/$SQN/init/group$g/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $S/$SQN/groups.json --group $g --out $S/$SQN/init/group$g 2>&1 | tail -1; say "slice group$g done"; }
done
# ---- square trainings (1 node each) ----
for g in 0 1 2 3; do
  TRAIN_SCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh launch_train $LOG/square${g}_launched $LOG/launch_square$g.log SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square $EXTRA_ENV
done
# ---- matched-point drivers (wait for checkpoints, merge, launch evals) ----
declare -a STEPS
for g in 0 1 2 3; do STEPS[$g]=$(python -c "
import json,math; t=json.load(open('$S/$SQN/pack/stats.json'))['tokens_per_group'][$g]; s=math.ceil(t/524288)
print(' '.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f' {s}')"); done
say "square steps: g0 [${STEPS[0]}] g1 [${STEPS[1]}] g2 [${STEPS[2]}] g3 [${STEPS[3]}]; token shares $SHARES"
for i in 0 1 2 3 4; do
  [ -f $LOG/merge_point$i.log ] || SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES SKIP_ORACLE=1 \
    STEPS_G0="${STEPS[0]}" STEPS_G1="${STEPS[1]}" STEPS_G2="${STEPS[2]}" STEPS_G3="${STEPS[3]}" bash scripts/sparse_experts/olmoe3_squares/merge_point.sh $i > $LOG/merge_point$i.log 2>&1 &
  [ -f $LOG/baseline_point$i.log ] || SQUARES_NAME=$SQN HELDOUT_DIR=$HR BASELINE_RUNS=$BRUN SKIP_ORACLE=1 bash scripts/sparse_experts/olmoe3_squares/baseline_point.sh $i > $LOG/baseline_point$i.log 2>&1 &
done
# ---- square passes + piecewise (needs the final square checkpoints and the start pass) ----
for g in 0 1 2 3; do last=${STEPS[$g]##* }; until [ -f $S/${RP}$g/step$last/train/rank0.pt ]; do sleep 300; done; done; say "all squares trained"
until [ -f $S/olmoe3_routing/$HR/$START/none/rank0/DONE ] || [ -f $S/olmoe3_routing/$HR/$START/none/meta.json ]; do sleep 120; done
[ -f $S/olmoe3_routing/$HR/$START/none/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $S/olmoe3_routing/$HR/$START/none 2>&1 | tail -1
POINTS="0 1 2 3 4" bash scripts/sparse_experts/olmoe3_squares/sub_passes.sh $TAG > $LOG/sub_passes.log 2>&1 &
# ---- finetune stage 1 (+0.5B): needs the 100% merge and the baseline's final checkpoint ----
until [ -f $S/$SQN/merged/match38148/merge_info.json ] && [ -f $S/$BRUN/step38147/train/rank0.pt ]; do sleep 300; done; say "100% merge and baseline final present"
S0=(${STEPS[0]}); S1=(${STEPS[1]}); S2=(${STEPS[2]}); S3=(${STEPS[3]})
[ -f $S/$SQN/merged_optim/match38148/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $S/$SQN/groups.json --full $S/$FULL/step19074 \
    --subs "$S/${RP}0/step${S0[4]},$S/${RP}1/step${S1[4]},$S/${RP}2/step${S2[4]},$S/${RP}3/step${S3[4]}" --weights "$SHARES" --out $S/$SQN/merged_optim/match38148 --with-optim --overwrite 2>&1 | tail -1
[ -f $S/$SQN/baseline_clean/step38147/model_and_optim/.metadata ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/$BRUN/step38147 --out $S/$SQN/baseline_clean/step38147 --overwrite 2>&1 | tail -1
for m in baseline merged; do
  src=$([ $m = merged ] && echo $S/$SQN/merged_optim/match38148 || echo $S/$SQN/baseline_clean/step38147)
  [ -f $S/$SQN/ft_start/${TAG}_$m/step38548/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $src --train-from $S/$BRUN/step38147 --start-step 38548 --out $S/$SQN/ft_start/${TAG}_$m 2>&1 | tail -1
  TRAIN_SCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh launch_train $LOG/ft_${m}_launched $LOG/launch_ft_$m.log FT_START=$W/$SQN/ft_start/${TAG}_$m FT_START_STEP=38548 FT_STEPS=954 OLMOE3_EMO=1 OLMOE3_RUNNAME=olmoe3_275m_${TAG}_${m}_ft OLMOE3_WANDB_TAGS=$SQN,finetune $EXTRA_ENV
done
for m in baseline merged; do RUN=olmoe3_275m_${TAG}_${m}_ft
  until [ -f $S/$RUN/step39502/train/rank0.pt ]; do sleep 180; done; say "$RUN finished"
  [ -f $S/olmoe3_routing/$HR/${m}_ft/none/rank0/DONE ] || [ -f $S/olmoe3_routing/$HR/${m}_ft/none/meta.json ] || launch "$SQN-eval-${m}_ft-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$RUN/step39502" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/${m}_ft/none/rank0" --restrict none --batch-size 8 --log-every 100
  [ -f $S/$SQN/ppl_validation/$RUN/step39502.json ] || launch "$SQN-eval-${m}_ft-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$RUN/step39502" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
done
for m in baseline merged; do R=$S/olmoe3_routing/$HR/${m}_ft/none; until [ -f $R/rank0/DONE ] || [ -f $R/meta.json ]; do sleep 120; done; [ -f $R/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $R 2>&1 | tail -1; done
# ---- finetune stage 2 (+1B) ----
bash scripts/sparse_experts/olmoe3_squares/ft2.sh $TAG > $LOG/ft2.log 2>&1
# ---- collect any remaining held-out passes ----
wait
for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "pipeline $TAG done"
