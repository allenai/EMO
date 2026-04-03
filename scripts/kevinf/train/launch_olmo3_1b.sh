#!/bin/bash
# Launch OLMo3-1B training on OLMoE-mix-0824
#
# Usage: bash src/scripts/kevinf/train/launch_olmo3_1b_v2.sh
#
# Model: OLMo3-1B (1B parameters)
# Data: OLMoE-mix-0824
# Training: 130B tokens on 2 nodes (16 GPUs)

runname="new-kevinf-olmo3-1b-130b-olmoemix-0824"

python -m olmo_core.launch.beaker \
  --name $runname \
  --gpus 8 \
  --nodes 2 \
  --weka=oe-training-default \
  --is_private_repo \
  --priority urgent \
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
  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
  --trainer.hard_stop='{value: 130_000_000_000, unit: tokens}' \
  --trainer.callbacks.downstream_evaluator.eval_interval=250 \
  --dataset.mix=OLMoE-mix-0824
