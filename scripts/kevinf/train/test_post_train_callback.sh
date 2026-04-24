#!/bin/bash
# Launch OLMo3-1B training on ChemPile dataset
#
# Usage: bash src/scripts/kevinf/train/launch_olmo3_1b_chempile.sh
#
# Model: OLMo3-1B trained to 130B on dolma3 from 150b subset
# Data: ChemPile (paper, education, lift)
runname="test-post-train-callback"

python -m olmo_core.launch.beaker \
  --name $runname \
  --gpus 8 \
  --nodes 1 \
  --weka=oe-training-default \
  --beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
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
  --trainer.max_duration='{value: 30, unit: steps}' \
  --trainer.hard_stop='{value: 30, unit: steps}' \
  --trainer.callbacks.downstream_evaluator.eval_interval=250 \
  --dataset.mix=chempile