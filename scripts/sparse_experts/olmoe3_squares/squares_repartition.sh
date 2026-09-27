#!/usr/bin/env bash
# Re-merge + re-partition every 10B (user request 2026-09-27): the EMO 512e random-partition control at sub-model LR 4e-4, from its
# 20B merge (the four window-1 squares of olmoe3_squares_emorand_lr4e-4 merged WITH Adam state, token-share weights) to 130B in eleven
# 10B cycles. Each cycle: a NEW random expert grouping of layers 2-9 (layer 1 whole, as everywhere else; seed 100+c), the merged model
# sliced into four squares, each square continuing on a contiguous quarter of the cycle's 10B of the training stream (a random quarter
# of its documents) at LR 4e-4 with a single checkpoint at its end, then merged with equal weights WITH Adam state (the next cycle's
# start) and evaluated on the 300B held-out sample and the v3-small ppl sets. Compared with the jointly trained baseline's existing
# points (30/35/61/87/113/130B; no baseline re-run, user decision).
# Storage (user decision 2026-09-27): squares + merged model kept only at the 30B, 60B, 90B, 120B and 130B cycle ends; at every other
# end both are deleted once the next cycle's squares have finished and the merge's evaluations are in. Idempotent; detach.
#   bash scripts/sparse_experts/olmoe3_squares/squares_repartition.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SRC=olmoe3_squares_emorand_lr4e-4; SRP=olmoe3_275m_emorand_lr4e-4_square
SQN=olmoe3_squares_emorand_lr4e-4_rp; SQ=$S/$SQN; RP=olmoe3_275m_emorand_lr4e-4_rp; HR=runs_heldout300b_emorand_lr4e-4_rp; FULL=olmoe3_275m_emo_10b; SAMPLE=sample_8k_300b.npz; LR=4e-4; K=4
TEMPLATE=$S/olmoe3_275m_30b_1node/step57221   # trainer state whose data-loader position make_finetune_start.py rewrites to any start step
ENDS=(38148 57221 76295 95368 114442 133515 152589 171662 190736 209809 228883 247956)   # 20B start, then every ~10B (19073.5 steps) to the 130B matched point
KEEP="57221 114442 171662 228883 247956"   # 30B, 60B, 90B, 120B, 130B: squares + merged model kept
LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR
say() { echo "$(date -u +%m-%d\ %H:%M) [rp] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; have $S/olmoe3_routing/$HR/$tag/none || launch "$SQN-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
merge_optim() { local groups=$1 out=$2 weights=$3; shift 3; [ -f $out/merge_info.json ] || { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $groups --full $S/$FULL/step19074 --subs "$(IFS=,; echo "$*")" --weights "$weights" --out $out --with-optim --overwrite 2>&1 | tail -1 | cut -c1-140; [ -f $out/config.json ] || cp $S/$FULL/step19074/config.json $out/config.json; }; }
keep() { case " $KEEP " in *" $1 "*) return 0;; *) return 1;; esac; }
cleanup() { local c=$1; local end=${ENDS[$c]}   # cycle c's artifacts, once cycle c+1's squares are done and merge c is evaluated (both checked by the caller)
  # (BUG until 2026-09-27 16:60: `local c=$1 end=${ENDS[$c]}` expanded $c BEFORE the assignment, i.e. with the caller's cycle -> the keep decision was shifted
  #  by one cycle: cycle 1 (30B, a keep point) was deleted and cycle 3 (50B) kept. Fixed; the retro pass below re-applies the rule to every finished cycle.)
  rm -rf $SQ/init_c$((c+1)) $SQ/ft_start_c$((c+1)); keep $end && { [ -d $SQ/merged_optim/c$c ] && say "cycle $c ($end): kept (squares + merged model)"; return 0; }
  for g in $(seq 0 $((K-1))); do rm -rf $S/${RP}_c${c}_sq$g; done; [ -d $SQ/merged_optim/c$c ] && { rm -rf $SQ/merged_optim/c$c; say "cycle $c ($end): squares + merged model deleted (not a keep point)"; }; return 0; }
evaluated() { have $S/olmoe3_routing/$HR/merged_c$1/none && [ -f $SQ/ppl_validation/merged_optim/c$1.json ]; }
# ---- cycle 0: the 4e-4 arm's 20B merge, with Adam state ----
if [ ! -f $SQ/merged_optim/c0/merge_info.json ]; then
  SH1=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$S/$SRC/pack/stats.json'))['token_share']))"); subs=()
  for g in 0 1 2 3; do t=$(python -c "import json; print(json.load(open('$S/$SRC/pack/stats.json'))['tokens_per_group'][$g])"); subs+=("$S/${SRP}$g/step$((t / 524288))"); done
  merge_optim $S/$SRC/groups.json $SQ/merged_optim/c0 "$SH1" "${subs[@]}"; say "cycle 0: 20B merge with Adam state done"
fi
for c in $(seq 1 11); do B=${ENDS[$((c-1))]}; E=${ENDS[$c]}; SQS=$(( (E - B + K - 1) / K )); G=$SQ/groups_c$c.json
  if [ -f $SQ/merged_optim/c$c/merge_info.json ] || evaluated $c; then   # a finished cycle (its squares may already be deleted by the storage rule): only its evaluations, never re-train
    [ -f $SQ/merged_optim/c$c/merge_info.json ] && { heldout merged_c$c $W/$SQN/merged_optim/c$c; ppl $W/$SQN/merged_optim/c$c merged_optim/c$c.json merged-c$c; }
    if [ $c -ge 2 ]; then p=$((c-1)); until evaluated $p; do sleep 300; done; for q in $(seq 1 $p); do evaluated $q && cleanup $q; done; fi
    continue
  fi
  [ -f $G ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $G --k $K --seed $((100 + c)) > /dev/null
  subs=(); for g in $(seq 0 $((K-1))); do start=$((B + g * SQS)); fin=$((start + SQS)); run=${RP}_c${c}_sq$g; subs+=("$S/$run/step$fin")
    [ -f $S/$run/step$fin/train/rank0.pt ] && continue
    [ -f $SQ/init_c$c/group$g/model_and_optim/.metadata ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $SQ/merged_optim/c$((c-1)) --groups $G --group $g --out $SQ/init_c$c/group$g/model_and_optim 2>&1 | tail -1 | cut -c1-120; say "cycle $c: slice group$g done"; }
    [ -f $SQ/ft_start_c$c/group$g/step$start/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/init_c$c/group$g --train-from $TEMPLATE --start-step $start --out $SQ/ft_start_c$c/group$g 2>&1 | tail -1 | cut -c1-120
    launch_train $LOG/c${c}_sq${g}_launched $LOG/launch_c${c}_sq$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_stdrand_square_w3.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_GROUPS=$W/$SQN/groups_c$c.json OLMOE3_EMO=1 OLMOE3_LR=$LR OLMOE3_RUNNAME=$run OLMOE3_WANDB_TAGS=$SQN,square,repartition,c$c FT_START=$W/$SQN/ft_start_c$c/group$g FT_START_STEP=$start FT_STEPS=$SQS OLMOE3_FIXED_STEPS=$fin
  done
  say "cycle $c: stream steps $B -> $E, $SQS per square"
  for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; done; done; say "cycle $c: square finals present"
  merge_optim $G $SQ/merged_optim/c$c 0.25,0.25,0.25,0.25 "${subs[@]}"
  heldout merged_c$c $W/$SQN/merged_optim/c$c; ppl $W/$SQN/merged_optim/c$c merged_optim/c$c.json merged-c$c
  if [ $c -ge 2 ]; then   # every earlier cycle that is evaluated and superseded (its successor's squares finished): apply the storage rule (idempotent retro pass)
    p=$((c-1)); until evaluated $p; do sleep 300; done
    for q in $(seq 1 $p); do evaluated $q && cleanup $q; done
  fi
done
until have $S/olmoe3_routing/$HR/merged_c11/none && [ -f $SQ/ppl_validation/merged_optim/c11.json ]; do sleep 300; done; rm -rf $SQ/init_c11 $SQ/ft_start_c11
for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
