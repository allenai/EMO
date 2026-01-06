#!/bin/bash
# Launch OLMo3-1B training on OLMo-mix-0625-150Bsample
#
# Usage: bash src/scripts/kevinf/train/launch_olmo3_1b.sh
#
# The data mix (OLMo-mix-0625-150Bsample) is hardcoded in OLMo3-1B.py
# Model: OLMo3-1B (1B parameters)
# Training: 150B tokens on 1 node (8 GPUs)

# runname="kevinf-olmo3-1b-130b-olmoemix-0824"
runname="kwargs-test"
python -m olmo_core.launch.beaker \
  --name $runname \
  --gpus 2 \
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
  train $runname ai2/jupiter \
  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
