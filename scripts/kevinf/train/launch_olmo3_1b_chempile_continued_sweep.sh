#!/bin/bash
# Launch OLMo3-1B continued pretraining on The Pile of Law
#
# Usage: bash src/scripts/kevinf/train/launch_olmo3_1b_the-pile-of-law_continued_pt_30B.sh
#
# Model: OLMo3-1B (1B parameters)
# Data: The Pile of Law (legal domain)
# Base: 130B Dolma pretrained checkpoint

# Hyperparameters (easy to change for sweeps)
dataset="chempile"
warmup_steps=715
train_tokens_B=10  # in billions
train_tokens_raw=$((train_tokens_B * 1000000000))
load_path="/weka/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"

# LR sweep: conservative (5e-5), moderate (1e-4), aggressive (2e-4)
for lr in 5e-5 1e-4 2e-4; do
  # Construct runname from hyperparams
  runname="olmo3-1b-${dataset}-${train_tokens_B}B-lr${lr}-warmup${warmup_steps}"
  if [ -n "$load_path" ]; then
    runname="${runname}-ctd"
  fi

  python -m olmo_core.launch.beaker \
    --name $runname \
    --gpus 8 \
    --nodes 1 \
    --weka=oe-training-default \
    --is_private_repo \
    --priority high \
    --shared-filesystem \
    --workspace ai2/flex2 \
    --cluster ai2/jupiter \
    --preemptible \
    --allow-dirty \
    --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" \
    -- src/scripts/kevinf/train/OLMo3-1B.py \
    $runname \
    --save-folder="/weka/oe-training-default/kevinf/checkpoints/${runname}/" \
    --work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
    --trainer.max_duration="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.hard_stop="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.callbacks.downstream_evaluator.eval_interval=250 \
    --dataset.mix=$dataset \
    --train_module.optim.lr=$lr \
    --train_module.scheduler.warmup_steps=$warmup_steps \
    ${load_path:+--load_path=$load_path} &

  sleep 5  # Brief pause between submissions
done

echo "All jobs submitted! Check Beaker for status."
