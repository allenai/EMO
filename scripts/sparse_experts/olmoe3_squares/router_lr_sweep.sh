#!/usr/bin/env bash
# Router-only finetune LR sweep (block A; user request 2026-09-18): from the 100% EMO merge, finetune ONLY the routed-expert
# routers for 0.5B tokens (steps 38548 -> 39502, same tokens as the full finetunes) at LRs 2e-4 / 2e-3 / 4e-3 / 8e-3 / 1.6e-2
# (8e-4 = the existing arm, router_ft.sh). Waits for the epoch-2-repaired start checkpoint (ft_start/emo_merged rebuilt by
# repair_epoch2/downstream.sh), checks every arm tensor by tensor (only routers changed), evaluates (held-out 20B-window CE +
# v3-small ppl).   bash scripts/sparse_experts/olmoe3_squares/router_lr_sweep.sh   (idempotent; detach it)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQ=$S/olmoe3_squares; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
LRS="${LRS:-2e-4 2e-3 4e-3 8e-3 1.6e-2}"; LOG=$SQ/logs; mkdir -p $LOG $SP
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
# ---- start checkpoint: the repaired 100% merge's hybrid finetune start (the pre-repair one was moved to *_epoch2) ----
until [ -d $SQ/ft_start/emo_merged_epoch2 ] && [ -f $SQ/ft_start/emo_merged/step38548/train/rank0.pt ] && [ -f $SQ/merged_optim/match38148/merge_info.json ]; do sleep 300; done
say "repaired start checkpoint present"
for lr in $LRS; do run=olmoe3_275m_emo_merged_rft_lr$lr
  [ -f $LOG/${run}_launched ] || [ -f $S/$run/step39502/train/rank0.pt ] && continue; u=""
  for attempt in 1 2 3 4 5 6; do FT_START="$W/olmoe3_squares/ft_start/emo_merged" FT_START_STEP=38548 FT_STEPS=954 OLMOE3_EMO=1 OLMOE3_ROUTER_ONLY_LR=$lr OLMOE3_RUNNAME=$run OLMOE3_WANDB_TAGS=olmoe3_squares,finetune,router_only,router_lr_sweep OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft_router.sh launch > $LOG/launch_$run.log 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/launch_$run.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$run: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/${run}_launched
done
for lr in $LRS; do run=olmoe3_275m_emo_merged_rft_lr$lr; tag=merged_rft_lr$lr
  until [ -f $S/$run/step39502/train/rank0.pt ]; do sleep 180; done; say "$run finished"
  [ -f $LOG/check_rft_lr$lr.json ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/check_router_only.py --start $SQ/merged_optim/match38148 --end $S/$run/step39502 --json $LOG/check_rft_lr$lr.json || say "ROUTER-ONLY CHECK FAILED for lr $lr (see $LOG/check_rft_lr$lr.json)"
  have $S/olmoe3_routing/runs_heldout20b/$tag/none || launch "olmoe3_squares-eval-$tag-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$run/step39502" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/runs_heldout20b/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100
  [ -f $SQ/ppl_validation/$run/step39502.json ] || launch "olmoe3_squares-eval-$tag-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$run/step39502" --out-dir "$W/olmoe3_squares/ppl_validation" --batch-size 8
done
for lr in $LRS; do R=$S/olmoe3_routing/runs_heldout20b/merged_rft_lr$lr/none; until have $R; do sleep 120; done; [ -f $R/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $R 2>&1 | tail -1; done
say "router LR sweep done"
