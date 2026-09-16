#!/usr/bin/env bash
# Router-recovery test for the EMO 512e squares (block A): from the 100% merged model, finetune ONLY the routed-expert
# routers for 0.5B tokens (stage 1, steps 38548 -> 39502) and then 0.5B more (stage 2, -> 40456), same tokens / LR as the
# full finetunes; verify after each stage that no other parameter changed (check_router_only.py); evaluate each stage
# (held-out 20B-window CE + v3-small ppl).   bash scripts/sparse_experts/olmoe3_squares/router_ft.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQ=$S/olmoe3_squares; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
LOG=$SQ/logs; mkdir -p $LOG
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
train() { local run=$1 start=$2 step=$3; [ -f $LOG/${run}_launched ] && return 0; local u=""
  for attempt in 1 2 3 4 5 6; do FT_START="$W/olmoe3_squares/ft_start/$start" FT_START_STEP=$step FT_STEPS=954 OLMOE3_EMO=1 OLMOE3_RUNNAME=$run OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft_router.sh launch > $LOG/launch_$run.log 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/launch_$run.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$run: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/${run}_launched; }
evals() { local run=$1 tag=$2 step=$3
  [ -f $S/olmoe3_routing/runs_heldout20b/$tag/none/rank0/DONE ] || [ -f $S/olmoe3_routing/runs_heldout20b/$tag/none/meta.json ] || launch "olmoe3_squares-eval-$tag-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$run/step$step" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/runs_heldout20b/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100
  [ -f $SQ/ppl_validation/$run/step$step.json ] || launch "olmoe3_squares-eval-$tag-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$run/step$step" --out-dir "$W/olmoe3_squares/ppl_validation" --batch-size 8; }
# ---- stage 1: from the 100% merge (hybrid start dir of the full merged finetune, weights = merged_optim/match38148) ----
train olmoe3_275m_emo_merged_rft emo_merged 38548
until [ -f $S/olmoe3_275m_emo_merged_rft/step39502/train/rank0.pt ]; do sleep 180; done; say "stage 1 finished"
PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/check_router_only.py --start $SQ/merged_optim/match38148 --end $S/olmoe3_275m_emo_merged_rft/step39502 --json $LOG/check_rft.json || { say "ROUTER-ONLY CHECK FAILED for stage 1; stopping"; exit 1; }
evals olmoe3_275m_emo_merged_rft merged_rft 39502
# ---- stage 2: continue from stage 1's checkpoint (clean rewrite, as for the full finetunes) ----
[ -f $SQ/ft2_clean/emo_merged_rft/model_and_optim/.metadata ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/olmoe3_275m_emo_merged_rft/step39502 --out $SQ/ft2_clean/emo_merged_rft --overwrite 2>&1 | tail -1
[ -f $SQ/ft_start/emo_merged_rft2/step39502/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/ft2_clean/emo_merged_rft --train-from $S/olmoe3_275m_emo_merged_rft/step39502 --start-step 39502 --out $SQ/ft_start/emo_merged_rft2 2>&1 | tail -1
train olmoe3_275m_emo_merged_rft2 emo_merged_rft2 39502
until [ -f $S/olmoe3_275m_emo_merged_rft2/step40456/train/rank0.pt ]; do sleep 180; done; say "stage 2 finished"
PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/check_router_only.py --start $SQ/ft2_clean/emo_merged_rft --end $S/olmoe3_275m_emo_merged_rft2/step40456 --json $LOG/check_rft2.json || { say "ROUTER-ONLY CHECK FAILED for stage 2; stopping"; exit 1; }
evals olmoe3_275m_emo_merged_rft2 merged_rft2 40456
for tag in merged_rft merged_rft2; do R=$S/olmoe3_routing/runs_heldout20b/$tag/none; until [ -f $R/rank0/DONE ] || [ -f $R/meta.json ]; do sleep 120; done; [ -f $R/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $R 2>&1 | tail -1; done
say "router-only finetunes done"
