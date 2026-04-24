#!/bin/bash
# Launch OLMo3-1B continued pretraining on medical-o1-reasoning-SFT
#
# Usage: bash scripts/kevinf/train/launch_medical_o1.sh
#
# Submits TWO jobs in parallel for ablation comparison:
#   - medical-o1-en-cot:   English Q&A + chain-of-thought reasoning (~12M tokens)
#   - medical-o1-en-nocot: English Q&A only, no CoT (~3.8M tokens)
#
# Model: OLMo3-1B (1B parameters)
# Base: 130B Dolma pretrained checkpoint
# Data: FreedomIntelligence/medical-o1-reasoning-SFT (en config)
# Note: Dataset is small (~12M tokens), model will cycle through many times

warmup_fraction=0.1
train_tokens_B=0.1  # in billions (adjust: ~83 epochs of en_cot, ~260 epochs of en_nocot)
train_tokens_raw=$(python3 -c "print(int(${train_tokens_B} * 1_000_000_000))")
load_path="/weka/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"
mix_base_dir="/weka/oe-training-default/ai2-llm"

for dataset in medical-o1-en-cot medical-o1-en-nocot; do
  for lr in 5e-5; do
    runname="olmo3-1b-${dataset}-${train_tokens_B}B-lr${lr}-warmup${warmup_fraction}"
    if [ -n "$load_path" ]; then
      runname="${runname}-ctd"
    fi

    echo "Submitting: $runname"

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
      --trainer.max_duration="{value: ${train_tokens_raw}, unit: tokens}" \
      --trainer.hard_stop="{value: ${train_tokens_raw}, unit: tokens}" \
      --trainer.callbacks.downstream_evaluator.eval_interval=100 \
      --dataset.mix=$dataset \
      --dataset.mix_base_dir=$mix_base_dir \
      --trainer.callbacks.lm_evaluator.eval_dataset.mix=$dataset \
      --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=$mix_base_dir \
      --trainer.callbacks.lm_evaluator.enabled=true \
      --train_module.optim.lr=$lr \
      ${load_path:+--load_path=$load_path} \
      --train_module.scheduler.warmup_fraction=$warmup_fraction \
      --data_loader.global_batch_size=2097152 &

    sleep 5
  done
done

wait
echo "All jobs submitted! Check Beaker for status."
