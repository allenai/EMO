#!/bin/bash
# Launch OLMo3-1B continued pretraining on MIMIC-IV-Note (clinical notes)
#
# Usage: bash scripts/kevinf/train/launch_mimic_iv_note.sh
#
# Model: OLMo3-1B (1B parameters)
# Data: MIMIC-IV-Note discharge (~1.0B tokens) + radiology (~0.7B tokens) = ~1.7B tokens
# Base: 130B Dolma pretrained checkpoint

# Hyperparameters (easy to change for sweeps)
dataset="mimic-iv-note"
warmup_fraction=0.1
train_tokens_B=5  # in billions (~3 epochs over 1.7B tokens)
train_tokens_raw=$((train_tokens_B * 1000000000))
load_path="/weka/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"

for lr in 5e-5; do
  # Construct runname from hyperparams
  runname="olmo3-1b-${dataset}-${train_tokens_B}B-lr${lr}-warmup${warmup_fraction}"
  if [ -n "$load_path" ]; then
    runname="${runname}-ctd"
  fi

  python -m olmo_core.launch.beaker \
    --name $runname \
    --gpus 8 \
    --nodes 1 \
    --weka=oe-training-default \
    --is_private_repo \
    --priority urgent \
    --shared-filesystem \
    --workspace ai2/flex2 \
    --cluster ai2/jupiter \
    --preemptible \
    --allow-dirty \
    --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY" \
    --env "S3_PROFILE=" \
    -- src/scripts/kevinf/train/OLMo3-1B.py \
    $runname \
    --save-folder="/weka/oe-training-default/kevinf/checkpoints/${runname}/" \
    --work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
    --trainer.max_duration="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.hard_stop="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.callbacks.downstream_evaluator.eval_interval=100 \
    --dataset.mix=$dataset \
    --dataset.mix_base_dir=s3://ai2-llm \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix=$dataset \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm \
    --trainer.callbacks.lm_evaluator.enabled=true \
    --trainer.callbacks.lm_evaluator.eval_interval=100 \
    --trainer.callbacks.lm_evaluator.eval_duration="{value: 200, unit: steps}" \
    --trainer.callbacks.lm_evaluator.eval_on_startup=true \
    --trainer.callbacks.lm_evaluator.log_interval=1 \
    --train_module.optim.lr=$lr \
    --train_module.scheduler.warmup_fraction=$warmup_fraction \
    ${load_path:+--load_path=$load_path} \

  sleep 5  # Brief pause between submissions
done

echo "All jobs submitted! Check Beaker for status."
