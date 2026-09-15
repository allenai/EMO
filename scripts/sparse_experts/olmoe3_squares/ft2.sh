#!/usr/bin/env bash
# Second post-merge finetuning stage (user request 2026-09-14): continue the finetuned merged model and the finetuned baseline
# from their step-39502 checkpoints for 0.5B more tokens (954 steps, 39502 -> 40456; 1B of joint finetuning in total), then
# evaluate (held-out 20B-window CE + v3-small ppl). The held-out sample is steps 38148-38547 of the stream; finetuning consumes
# steps 38548-40456, so it stays unseen. Start checkpoints are rewritten to the clean single-file layout (no SkipStepAdamW
# windows) exactly like the first stage.
#   bash scripts/sparse_experts/olmoe3_squares/ft2.sh [emo|std|k8]     (k8 shares the EMO baseline: merged model only)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
V="${1:-emo}"
case $V in
  emo) PFX=olmoe3_275m_emo_; HR=runs_heldout20b; SQN=olmoe3_squares; EMO=1; MODELS="baseline merged";;
  std) PFX=olmoe3_275m_; HR=runs_heldout20b_std; SQN=olmoe3_squares_std; EMO=0; MODELS="baseline merged";;
  k8)  PFX=olmoe3_275m_emo_k8_; HR=runs_heldout20b_k8; SQN=olmoe3_squares_k8; EMO=1; MODELS="merged";;
  pool64or512) PFX=olmoe3_275m_pool64or512_; HR=runs_heldout20b_pool64or512; SQN=olmoe3_squares_pool64or512; EMO=1; MODELS="baseline merged"; export OLMOE3_EMO_POOL_DIST=choice:64,512;;
  learnedd) PFX=olmoe3_275m_learnedd_; HR=runs_heldout20b_learnedd; SQN=olmoe3_squares_learnedd; EMO=1; MODELS="baseline merged"
       export OLMOE3_EMO_LEARNED_D=1 OLMOE3_LD_SIGNAL=coverage OLMOE3_LD_LAMBDA=1.0 OLMOE3_LD_LAMBDA_COV=1.0 OLMOE3_LD_TEMP=2.0 OLMOE3_LD_WARMUP=0 OLMOE3_LD_LR_MULT=10;;
  *) echo "emo|std|k8|pool64or512|learnedd"; exit 1;;
esac
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
START=39502; STEPS=954; END=$((START + STEPS))
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  echo "$(date -u +%H:%M) $name: ${u:-LAUNCH FAILED}"; }
for m in $MODELS; do
  SRC=$S/${PFX}${m}_ft/step$START; CLEAN=$S/$SQN/ft2_clean/$m; FS=$S/$SQN/ft_start/${m}_ft2; RUN=${PFX}${m}_ft2
  if [ -f "$S/$RUN/step$END/train/rank0.pt" ]; then echo "== $RUN already trained"; continue; fi
  [ -f "$CLEAN/model_and_optim/.metadata" ] || { echo "$(date -u +%H:%M) rewriting $SRC"; PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src "$SRC" --out "$CLEAN" --overwrite 2>&1 | tail -1; }
  [ -f "$FS/step$START/train/rank0.pt" ] || { PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model "$CLEAN" --train-from "$SRC" --start-step $START --out "$FS" 2>&1 | tail -1; }
  for attempt in 1 2 3 4 5 6; do L=$SP/launch_$RUN.log
    FT_START="$W/$SQN/ft_start/${m}_ft2" FT_START_STEP=$START FT_STEPS=$STEPS OLMOE3_EMO=$EMO OLMOE3_WANDB_TAGS="$SQN,finetune" OLMOE3_RUNNAME=$RUN OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh launch > "$L" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$L" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  echo "$(date -u +%H:%M) $RUN: ${u:-LAUNCH FAILED}"
done
for m in $MODELS; do RUN=${PFX}${m}_ft2
  until [ -f "$S/$RUN/step$END/train/rank0.pt" ]; do sleep 180; done; echo "$(date -u +%H:%M) $RUN finished"
  [ -f "$S/olmoe3_routing/$HR/${m}_ft2/none/rank0/DONE" ] || launch "$SQN-eval-${m}_ft2-none" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/$RUN/step$END" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$HR/${m}_ft2/none/rank0" --restrict none --batch-size 8 --log-every 100
  [ -f "$S/$SQN/ppl_validation/$RUN/step$END.json" ] || launch "$SQN-eval-${m}_ft2-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$RUN/step$END" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
done
for m in $MODELS; do R=$S/olmoe3_routing/$HR/${m}_ft2/none
  until [ -f "$R/rank0/DONE" ]; do sleep 120; done; python scripts/sparse_experts/olmoe3_routing/merge_routing.py "$R" 2>&1 | tail -1
done
echo "$(date -u +%H:%M) ft2 $V evals done"
