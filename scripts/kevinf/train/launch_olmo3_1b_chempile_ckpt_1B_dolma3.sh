#!/bin/bash
# Launch OLMo3-1B training on ChemPile dataset
#
# Usage: bash src/scripts/kevinf/train/launch_olmo3_1b_chempile.sh
#
# Model: OLMo3-1B trained to 130B on dolma3 from 150b subset
# Data: ChemPile (paper, education, lift)
runname="olmo3-1b-10B-chempile-papers_education_lift-ckpt_1B_dolma3"

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
  --trainer.max_duration='{value: 10_000_000_000, unit: tokens}' \
  --trainer.hard_stop='{value: 10_000_000_000, unit: tokens}' \
  --trainer.callbacks.downstream_evaluator.eval_interval=250 \
  --dataset.mix=chempile \
  --load_path="/weka/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"