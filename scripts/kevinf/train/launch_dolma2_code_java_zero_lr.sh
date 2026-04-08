#!/bin/bash
# Zero-LR diagnostic: verify LM evaluator determinism
#
# Usage: bash scripts/kevinf/train/launch_dolma2_code_java_zero_lr.sh
#
# With lr=0, model weights never change, so every eval pass should produce
# the EXACT same perplexity. If values differ -> non-determinism bug.
# If values are identical -> eval is correct, bounciness is from weight changes.
#
# Evals run every training step (eval_interval=1) for fast verification.

dataset="dolma2-code-java"
warmup_fraction=0.1
train_tokens_B=1  # short run, enough for many eval passes
train_tokens_raw=$((train_tokens_B * 1000000000))
load_path="/weka/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"
lr=0

runname="train-olmo3-1b-${dataset}-zero-lr-diagnostic-2gpu-post-reset-fix-no-epoch"
if [ -n "$load_path" ]; then
  runname="${runname}-ctd"
fi

python -m olmo_core.launch.beaker \
  --name $runname \
  --gpus 2 \
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
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --dataset.mix=$dataset \
  --dataset.mix_base_dir=s3://ai2-llm \
  --trainer.callbacks.lm_evaluator.eval_dataset.mix=dolma2-code-java \
  --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=s3://ai2-llm \
  --trainer.callbacks.lm_evaluator.enabled=true \
  --trainer.callbacks.lm_evaluator.eval_interval=1 \
  --trainer.callbacks.lm_evaluator.eval_on_startup=true \
  --train_module.optim.lr=$lr \
  ${load_path:+--load_path=$load_path} \
  --train_module.scheduler.warmup_fraction=$warmup_fraction

echo "Zero-LR diagnostic job submitted! Check Beaker for status."
echo "Expected: every eval pass should produce identical PPL values."
