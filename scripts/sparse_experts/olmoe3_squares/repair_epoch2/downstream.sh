#!/usr/bin/env bash
# After launch.sh: wait for the repaired finals, move every artifact derived from the tainted finals aside (suffix
# _epoch2 / .epoch2.json, nothing deleted) and redo it: the 100% merge with held-out + v3-small evals, the square
# passes + piecewise, the post-merge finetunes (+0.5B, +1B; merged model only -- the baseline finetunes never touched a
# square), the router-only finetune (block A) and the standard-MoE window 2 (retrained from the repaired window-1
# finals). Idempotent (reuses what exists); variants run in parallel; detach it and follow <squares dir>/logs/repair_downstream.log.
#   bash scripts/sparse_experts/olmoe3_squares/repair_epoch2/downstream.sh [variant ...]
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
source scripts/sparse_experts/olmoe3_squares/repair_epoch2/variants.sh
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; OUT=claude_outputs/olmoe3_routing
MIN_FREE_GB="${MIN_FREE_GB:-300}"; mkdir -p $SP
say() { echo "$(date -u +%m-%d\ %H:%M) [$V] $*"; }
# only PRE-repair artifacts are set aside (older than the repair launch, 2026-09-18 04:10 UTC): the *_epoch2 copies are purged
# (purge.sh), so their absence must not make a re-run move the repaired outputs aside
REPAIR_T0="2026-09-18 04:10 UTC"
old() { [ -e "$1" ] && [ -z "$(find "$1" -maxdepth 0 -newermt "$REPAIR_T0" 2>/dev/null)" ]; }
aside() { local p; for p in "$@"; do old "$p" && [ ! -e "${p}_epoch2" ] && mv "$p" "${p}_epoch2" && say "aside $p"; done; return 0; }
aside_json() { local p; for p in "$@"; do old "$p" && [ ! -f "${p%.json}.epoch2.json" ] && mv "$p" "${p%.json}.epoch2.json" && say "aside $p"; done; return 0; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
free_gb() { df --output=avail -BG /root/EMO | tail -1 | tr -dc 0-9; }
wait_free() { while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do say "weka has $(free_gb)G free (< ${MIN_FREE_GB}G); waiting -- delete *_epoch2 artifacts to proceed"; sleep 900; done; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; local R=$S/olmoe3_routing/$HR/$tag/none; have $R || launch "$SQN-repair-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $S/$SQN/ppl_validation/$json ] || launch "$SQN-repair-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
finish() { local R=$1; until have $R; do sleep 120; done; [ -f $R/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $R 2>&1 | tail -1; }

repair_variant() {
  V=$1; variant_env $V || return 1; SQ=$S/$SQN; LOG=$SQ/logs; mkdir -p $LOG
  local specs finals g st subs u RUN
  specs=($(variant_specs $SQN $RP $PACK $K)) || { say "spec computation failed"; return 1; }
  SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/$PACK/stats.json'))['token_share']))")
  finals=(); for spec in "${specs[@]}"; do IFS=: read -r g tokens s old resume <<< "$spec"; finals+=($s); done
  # ---- 1. repaired finals ----
  for g in $(seq 0 $((K-1))); do until [ -f $S/${RP}$g/step${finals[$g]}/train/rank0.pt ]; do sleep 300; done; done; say "repaired finals present: ${finals[*]}"
  # ---- 2. tainted derived artifacts aside ----
  aside $SQ/merged/match38148 $SQ/merged_optim/match38148 $S/olmoe3_routing/$HR/merged_match38148 $SQ/ft_start/${FTTAG}_merged $SQ/ft_start/merged_ft2 $SQ/ft2_clean/merged \
        $S/${PFX}merged_ft $S/${PFX}merged_ft2 $S/olmoe3_routing/$HR/merged_ft $S/olmoe3_routing/$HR/merged_ft2 $SQ/ppl_validation/${PFX}merged_ft $SQ/ppl_validation/${PFX}merged_ft2 $LOG/repair_ft_launched
  for g in $(seq 0 $((K-1))); do aside $S/olmoe3_routing/$HR/sub${g}_match38148; done
  aside_json $SQ/ppl_validation/merged/match38148.json $OUT/$SQO/piecewise_match38148.json
  if [ $V = emo ]; then
    aside $S/olmoe3_275m_emo_merged_rft $S/olmoe3_275m_emo_merged_rft2 $SQ/ft_start/emo_merged_rft2 $SQ/ft2_clean/emo_merged_rft $S/olmoe3_routing/$HR/merged_rft $S/olmoe3_routing/$HR/merged_rft2 \
          $SQ/ppl_validation/olmoe3_275m_emo_merged_rft $SQ/ppl_validation/olmoe3_275m_emo_merged_rft2 $LOG/olmoe3_275m_emo_merged_rft_launched $LOG/olmoe3_275m_emo_merged_rft2_launched $LOG/check_rft.json $LOG/check_rft2.json
  fi
  if [ $V = std ]; then local HR3=runs_heldout300b_std
    aside $SQ/init2 $S/olmoe3_routing/$HR3/merged_match38148; aside_json $OUT/olmoe3_squares_std_w2/piecewise_match38148.json
    for g in 0 1 2 3; do aside $S/olmoe3_275m_square${g}_w2 $S/olmoe3_routing/$HR3/sub${g}_match38148 $LOG/square${g}_w2_launched; done
    for st in 39073 44073 49073 54073 57221; do aside $SQ/merged/match$st $S/olmoe3_routing/$HR3/merged_match$st; aside_json $SQ/ppl_validation/merged/match$st.json $OUT/olmoe3_squares_std_w2/piecewise_match$st.json
      for g in 0 1 2 3; do aside $S/olmoe3_routing/$HR3/sub${g}_match$st; done; done
  fi
  # ---- 3. 100% merge + its held-out / v3-small evals (merge_point.sh merges synchronously, launches the evals) ----
  for g in $(seq 0 $((K-1))); do st=$(fixed_steps ${finals[$g]}); export STEPS_G$g="${st//,/ }"; done
  [ -f $SQ/merged/match38148/merge_info.json ] && [ -f $SQ/ppl_validation/merged/match38148.json ] || \
    SQUARES_NAME=$SQN SQUARE_RUN_PREFIX=$RP FULL_RUN=$FULL HELDOUT_DIR=$HR MERGE_WEIGHTS=$SHARES SKIP_ORACLE=1 K=$K bash scripts/sparse_experts/olmoe3_squares/merge_point.sh 4 2>&1 | sed "s/^/[$V merge_point] /"
  say "100% merge done"
  # ---- 4. square passes + piecewise (background) ----
  if [ $HAS_SUB = 1 ]; then ( POINTS=4 bash scripts/sparse_experts/olmoe3_squares/sub_passes.sh $V > $LOG/repair_sub_passes.log 2>&1
      finish $S/olmoe3_routing/$HR/merged_match38148/none; HELDOUT_DIR=$HR SQUARES_NAME=$SQN START_TAG=$START python scripts/sparse_experts/olmoe3_squares/piecewise_eval.py --name match38148 | head -2; say "piecewise match38148 done" ) &
  fi
  # ---- 4b. standard MoE window 2 (background): re-evaluates window 1 on the 300B sample, retrains the squares 20B -> 30B ----
  if [ $V = std ]; then ( wait_free; bash scripts/sparse_experts/olmoe3_squares/squares_window2_std.sh > $LOG/repair_window2.log 2>&1; say "window 2 done" ) & fi
  # ---- 5. post-merge finetune +0.5B of the merged model ----
  wait_free
  subs=""; for g in $(seq 0 $((K-1))); do subs="$subs,$S/${RP}$g/step${finals[$g]}"; done; subs=${subs#,}
  [ -f $SQ/merged_optim/match38148/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$subs" --weights "$SHARES" --out $SQ/merged_optim/match38148 --with-optim --overwrite 2>&1 | tail -1 | cut -c1-120
  [ -f $SQ/ft_start/${FTTAG}_merged/step38548/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/merged_optim/match38148 --train-from $S/$BRUN/step38147 --start-step 38548 --out $SQ/ft_start/${FTTAG}_merged 2>&1 | tail -1
  RUN=${PFX}merged_ft
  if [ ! -f $S/$RUN/step39502/train/rank0.pt ] && [ ! -f $LOG/repair_ft_launched ]; then u=""
    for attempt in 1 2 3 4 5 6; do env FT_START=$W/$SQN/ft_start/${FTTAG}_merged FT_START_STEP=38548 FT_STEPS=954 OLMOE3_EMO=$EMO OLMOE3_RUNNAME=$RUN OLMOE3_WANDB_TAGS=$SQN,finetune,epoch2_repair $EXTRA_ENV OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh launch > $LOG/repair_launch_ft.log 2>&1
      u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/repair_launch_ft.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
    say "$RUN: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/repair_ft_launched
  fi
  until [ -f $S/$RUN/step39502/train/rank0.pt ]; do sleep 180; done; say "$RUN finished"
  heldout merged_ft $W/$RUN/step39502; ppl $W/$RUN/step39502 $RUN/step39502.json merged_ft
  # ---- 6. router-only finetune (block A; needs ft_start/emo_merged) and the +1B stage ----
  if [ $V = emo ]; then ( bash scripts/sparse_experts/olmoe3_squares/router_ft.sh > $LOG/repair_router_ft.log 2>&1; say "router-only finetune done" ) & fi
  if [ $HAS_FT2 = 1 ]; then wait_free; bash scripts/sparse_experts/olmoe3_squares/ft2.sh $V > $LOG/repair_ft2.log 2>&1; say "ft2 done"; fi
  finish $S/olmoe3_routing/$HR/merged_ft/none
  wait; say "variant done"
}
VARIANTS="${*:-emo noemo std k8 pool64or512 learnedd}"
for V in $VARIANTS; do sqn=$(variant_env $V && echo $SQN) || exit 1; mkdir -p $S/$sqn/logs; repair_variant $V >> $S/$sqn/logs/repair_downstream.log 2>&1 & done
wait; echo "$(date -u +%m-%d\ %H:%M) all variants done"
