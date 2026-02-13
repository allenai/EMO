#!/bin/bash
# Launch OLMo3-1B continued pretraining on Java code_fim
#
# Usage: bash scripts/kevinf/train/launch_code_fim_java.sh
#
# Model: OLMo3-1B (1B parameters)
# Data: code_fim Java subset
# Eval: code_fim_by_lang (per-language PPL)

# Hyperparameters
dataset="code_fim_java"
eval_dataset="code_fim_by_lang"
warmup_fraction=0.1
train_tokens_B=2  # in billions
train_tokens_raw=$((train_tokens_B * 1000000000))
load_path="/weka/oe-training-default/kevinf/checkpoints-new/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995"

for lr in 5e-5 ; do
  runname="olmo3-1b-${dataset}-${train_tokens_B}B-lr${lr}-warmup${warmup_fraction}-pplx-raw"
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
    --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" \
    -- src/scripts/kevinf/train/OLMo3-1B.py \
    $runname \
    --save-folder="/weka/oe-training-default/kevinf/checkpoints/${runname}/" \
    --work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
    --trainer.max_duration="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.hard_stop="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.callbacks.downstream_evaluator.eval_interval=100 \
    --trainer.callbacks.lm_evaluator.eval_dataset.mix=$eval_dataset \
    --dataset.mix=$dataset \
    --eval-mix $eval_dataset \
    --train_module.optim.lr=$lr \
    ${load_path:+--load_path=$load_path}
    # --train_module.scheduler.warmup_fraction=$warmup_fraction \

done
