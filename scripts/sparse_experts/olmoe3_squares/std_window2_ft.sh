#!/usr/bin/env bash
# Joint finetuning of the standard-MoE merges at BOTH window ends (block B; user request 2026-09-18), all evaluated on the
# 300B-token held-out sample: (a) 20B: the repaired merged / baseline finetunes (+0.5B steps 38548->39502, +1B ->40456, from
# repair_epoch2 / ft2.sh) only need the 300B-sample passes; (b) 30B: merge the repaired window-2 finals with Adam state,
# finetune merged and baseline from step 57221 for 0.5B (-> 58175) and 1B (-> 59129) tokens of the stream right after 30B
# (unseen by every model; the 300B sample is 515k steps later), then held-out + v3-small evals.
#   bash scripts/sparse_experts/olmoe3_squares/std_window2_ft.sh   (idempotent; detach it)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SQN=olmoe3_squares_std; SQ=$S/$SQN; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
HR=runs_heldout300b_std; SAMPLE=olmoe3_routing/sample_8k_300b.npz; FULL=olmoe3_275m_10b; BRUN=olmoe3_275m_30b_1node; LOG=$SQ/logs; mkdir -p $LOG $SP
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; local R=$S/olmoe3_routing/$HR/$tag/none; have $R || launch "$SQN-w2ft-eval-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-w2ft-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
train() { local run=$1 start=$2 step=$3; [ -f $LOG/${run}_launched ] || [ -f $S/$run/step$((step + 954))/train/rank0.pt ] && return 0; local u=""
  for attempt in 1 2 3 4 5 6; do FT_START="$W/$SQN/ft_start/$start" FT_START_STEP=$step FT_STEPS=954 OLMOE3_EMO=0 OLMOE3_RUNNAME=$run OLMOE3_WANDB_TAGS=$SQN,finetune,w2 OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh launch > $LOG/launch_$run.log 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/launch_$run.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$run: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $LOG/${run}_launched; }
# ---- (a) 20B: repaired merged / baseline finetunes on the 300B sample ----
repaired() { [ -n "$(find "$1" -maxdepth 0 -newermt "2026-09-18 04:10 UTC" 2>/dev/null)" ]; }   # written after the epoch-2 repair launch (the pre-repair copies are purged, not kept)
until [ -f $S/olmoe3_275m_merged_ft/step39502/train/rank0.pt ] && repaired $S/olmoe3_275m_merged_ft/step39502/train/rank0.pt; do sleep 300; done
heldout merged_ft $W/olmoe3_275m_merged_ft/step39502; heldout baseline_ft $W/olmoe3_275m_baseline_ft/step39502; heldout baseline_ft2 $W/olmoe3_275m_baseline_ft2/step40456
until [ -f $S/olmoe3_275m_merged_ft2/step40456/train/rank0.pt ] && repaired $S/olmoe3_275m_merged_ft2/step40456/train/rank0.pt; do sleep 300; done
heldout merged_ft2 $W/olmoe3_275m_merged_ft2/step40456; say "20B finetune passes launched"
# ---- (b) 30B: repaired window-2 finals (floor steps of pack2) -> merge with Adam state -> finetune start; baseline 30B start ----
W2F=(); for g in 0 1 2 3; do W2F+=($(python -c "import json; print(json.load(open('$SQ/pack2/stats.json'))['tokens_per_group'][$g] // 524288)")); done
SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack2/stats.json'))['token_share']))")
for g in 0 1 2 3; do until [ -f $S/olmoe3_275m_square${g}_w2/step${W2F[$g]}/train/rank0.pt ]; do sleep 300; done; done; say "repaired window-2 finals present: ${W2F[*]}"
until [ -f $S/$BRUN/step57221/train/rank0.pt ]; do sleep 300; done
subs="$S/olmoe3_275m_square0_w2/step${W2F[0]},$S/olmoe3_275m_square1_w2/step${W2F[1]},$S/olmoe3_275m_square2_w2/step${W2F[2]},$S/olmoe3_275m_square3_w2/step${W2F[3]}"
[ -f $SQ/merged_optim/match57221/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$subs" --weights "$SHARES" --out $SQ/merged_optim/match57221 --with-optim --overwrite 2>&1 | tail -1 | cut -c1-120
[ -f $SQ/baseline_clean/step57221/model_and_optim/.metadata ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/$BRUN/step57221 --out $SQ/baseline_clean/step57221 --overwrite 2>&1 | tail -1
for m in merged baseline; do src=$([ $m = merged ] && echo $SQ/merged_optim/match57221 || echo $SQ/baseline_clean/step57221)
  [ -f $SQ/ft_start/std_${m}30/step57221/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $src --train-from $S/$BRUN/step57221 --start-step 57221 --out $SQ/ft_start/std_${m}30 2>&1 | tail -1
  train olmoe3_275m_${m}30_ft std_${m}30 57221
done
for m in merged baseline; do run=olmoe3_275m_${m}30_ft; until [ -f $S/$run/step58175/train/rank0.pt ]; do sleep 180; done; say "$run finished"
  heldout ${m}30_ft $W/$run/step58175; ppl $W/$run/step58175 $run/step58175.json ${m}30_ft
  [ -f $SQ/ft2_clean/${m}30/model_and_optim/.metadata ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/$run/step58175 --out $SQ/ft2_clean/${m}30 --overwrite 2>&1 | tail -1
  [ -f $SQ/ft_start/std_${m}30_ft2/step58175/train/rank0.pt ] || PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/make_finetune_start.py --model $SQ/ft2_clean/${m}30 --train-from $S/$run/step58175 --start-step 58175 --out $SQ/ft_start/std_${m}30_ft2 2>&1 | tail -1
  train olmoe3_275m_${m}30_ft2 std_${m}30_ft2 58175
done
for m in merged baseline; do run=olmoe3_275m_${m}30_ft2; until [ -f $S/$run/step59129/train/rank0.pt ]; do sleep 180; done; say "$run finished"
  heldout ${m}30_ft2 $W/$run/step59129; ppl $W/$run/step59129 $run/step59129.json ${m}30_ft2
done
for tag in merged_ft merged_ft2 baseline_ft baseline_ft2 merged30_ft merged30_ft2 baseline30_ft baseline30_ft2; do R=$S/olmoe3_routing/$HR/$tag/none; until have $R; do sleep 120; done; [ -f $R/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $R 2>&1 | tail -1; done
say "window-2 joint finetunes done"
