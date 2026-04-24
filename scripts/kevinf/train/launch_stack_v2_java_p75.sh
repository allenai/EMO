#!/bin/bash
# Launch OLMo3-1B continued pretraining on stack-v2-java-p75 (Stack-v2 Java quality_p75+)
#
# Usage: bash scripts/kevinf/train/launch_stack_v2_java_p75.sh
#
# Model: OLMo3-1B (1B parameters)
# Data: stack-v2-java-p75 (Stack-v2 Java quality_p75 through p95, ~24.8B tokens available)
# Training: capped at 10B tokens via max_duration
# Base: 130B Dolma pretrained checkpoint

# Hyperparameters (easy to change for sweeps)
dataset="stack-v2-java-p75"
warmup_fraction=0.1
train_tokens_B=10  # in billions
train_tokens_raw=$((train_tokens_B * 1000000000))
load_path="/weka/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"

for lr in 5e-5; do
  runname="train-olmo3-1b-${dataset}-${train_tokens_B}B-lr${lr}"
  if [ -n "$load_path" ]; then
    runname="${runname}-ctd"
  fi

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
    --trainer.callbacks.lm_evaluator.eval_dataset.mix=stack-v2-all-p75 \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm \
    --trainer.callbacks.lm_evaluator.name=lm_stack_v2_all \
    --trainer.callbacks.lm_evaluator.enabled=true \
    --trainer.callbacks.lm_evaluator_2.eval_dataset.mix=stack-v2-java-p75 \
    --trainer.callbacks.lm_evaluator_2.eval_dataset.mix_base_dir=s3://ai2-llm \
    --trainer.callbacks.lm_evaluator_2.name=lm_java \
    --trainer.callbacks.lm_evaluator_2.enabled=true \
    --train_module.optim.lr=$lr \
    ${load_path:+--load_path=$load_path} \
    --train_module.scheduler.warmup_fraction=$warmup_fraction

  sleep 5
done

echo "All jobs submitted! Check Beaker for status."
