#!/usr/bin/env bash
# One-off recovery (2026-09-18 ~09:40 UTC): weka hit 100% around 07:10-07:20 UTC and killed the k8 / pool64or512 / learnedd
# merged finetunes, the standard +1B stage, four router-LR sweep arms and three held-out passes ("No space left on device").
# Relaunch them; the waiting drivers (downstream.sh, ft2.sh, router_lr_sweep.sh) only poll for the output files.
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
source scripts/sparse_experts/olmoe3_squares/repair_epoch2/variants.sh
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
train() { local script=$1 run=$2 start=$3 step=$4; shift 4; local u=""   # remaining args: extra env
  rm -rf $S/$run   # partial checkpoints of the crashed attempt
  for attempt in 1 2 3 4 5 6; do env FT_START="$start" FT_START_STEP=$step FT_STEPS=954 OLMOE3_RUNNAME=$run OLMOE3_FOLLOW=0 "$@" bash scripts/sparse_experts/model_scripts/$script launch > $SP/launch_$run.log 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' $SP/launch_$run.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$run: ${u:-LAUNCH FAILED}"; }
# merged finetunes (+0.5B) of k8 / pool64or512 / learnedd
for V in k8 pool64or512 learnedd; do variant_env $V
  [ -f $S/${PFX}merged_ft/step39502/train/rank0.pt ] || train olmoe3_275m_ft.sh ${PFX}merged_ft $W/$SQN/ft_start/${FTTAG}_merged 38548 OLMOE3_EMO=$EMO OLMOE3_WANDB_TAGS=$SQN,finetune,epoch2_repair $EXTRA_ENV
done
# standard +1B stage
[ -f $S/olmoe3_275m_merged_ft2/step40456/train/rank0.pt ] || train olmoe3_275m_ft.sh olmoe3_275m_merged_ft2 $W/olmoe3_squares_std/ft_start/merged_ft2 39502 OLMOE3_EMO=0 OLMOE3_WANDB_TAGS=olmoe3_squares_std,finetune
# router-LR sweep arms
for lr in 2e-3 4e-3 8e-3 1.6e-2; do run=olmoe3_275m_emo_merged_rft_lr$lr
  [ -f $S/$run/step39502/train/rank0.pt ] || train olmoe3_275m_ft_router.sh $run $W/olmoe3_squares/ft_start/emo_merged 38548 OLMOE3_EMO=1 OLMOE3_ROUTER_ONLY_LR=$lr OLMOE3_WANDB_TAGS=olmoe3_squares,finetune,router_only,router_lr_sweep
done
# held-out passes that died
pass() { local hr=$1 tag=$2 ckpt=$3; [ -f $S/olmoe3_routing/$hr/$tag/none/meta.json ] || [ -f $S/olmoe3_routing/$hr/$tag/none/rank0/DONE ] || launch "relaunch-$hr-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/sample_8k_20b.npz" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
pass runs_heldout20b merged_ft $W/olmoe3_275m_emo_merged_ft/step39502
pass runs_heldout20b_noemo merged_ft $W/olmoe3_275m_noemo_merged_ft/step39502
pass runs_heldout20b merged_rft_lr2e-4 $W/olmoe3_275m_emo_merged_rft_lr2e-4/step39502
say "relaunch done"
