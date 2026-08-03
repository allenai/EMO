#!/usr/bin/env bash
# Run one dense-1B Step-0 scheduler endpoint on the verified 0802 Pool-A corpus.
# Cosine endpoints always start fresh. The WSD curve is supplied by the matching
# Step-2 Pool-A chain and should not be duplicated with this launcher.
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
BEAKER_AUTO_RESUME="${BEAKER_AUTO_RESUME:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

scheduler="${SCHEDULER:?Set SCHEDULER to cosine or cosine0}"
epochs="${EPOCHS:?Set EPOCHS from 1 through 5}"
lr="${LR:?Set LR}"
wd="${WD:-0.033}"
subset_manifest="${SUBSET_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json}"
validation_manifest="${VALIDATION_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_validation.json}"
init_seed="${INIT_SEED:-12536}"
data_seed="${DATA_SEED:-0}"

case "${lr}" in 2.5e-4|5e-4|1e-3|2e-3|4e-3) ;; *) echo "Unsupported LR: ${lr}" >&2; exit 2 ;; esac
case "${epochs}" in
  1) max_tokens=1000000000; checkpoint_steps='[239]' ;;
  2) max_tokens=2000000000; checkpoint_steps='[239,477]' ;;
  3) max_tokens=3000000000; checkpoint_steps='[239,477,716]' ;;
  4) max_tokens=4000000000; checkpoint_steps='[239,477,716,954]' ;;
  5) max_tokens=5000000000; checkpoint_steps='[239,477,716,954,1193]' ;;
  *) echo "EPOCHS must be 1 through 5" >&2; exit 2 ;;
esac
for manifest in "${subset_manifest}" "${validation_manifest}"; do
  [[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 2; }
done
case "${scheduler}" in
  cosine) alpha_f=0.1 ;;
  cosine0) alpha_f=0 ;;
  *) echo "SCHEDULER must be cosine or cosine0" >&2; exit 2 ;;
esac

warmup_steps=24
downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
scheduler_config="{_CLASS_: olmo_core.optim.scheduler.CosWithWarmup, units: steps, warmup: ${warmup_steps}, alpha_f: ${alpha_f}}"
runname="dense_1b_step0_0802_repeated_dclm1b_${scheduler}_e${epochs}_lr${lr}_wd${wd}_warmup${warmup_steps}"

launch src/scripts/train/olmo2-1B.py "${runname}" \
  --save-folder="${MODELS_DIR}/${runname}" \
  --dataset.mix=null \
  --dataset.subset_manifest="${subset_manifest}" \
  --dataset.mix_base_dir="${DATA_ROOT}" \
  --work-dir="${DATASET_CACHE}" \
  --trainer.max_duration="{value: ${max_tokens}, unit: tokens}" \
  --trainer.callbacks.wandb.enabled=true \
  --trainer.callbacks.wandb.entity=ai2-llm \
  --trainer.callbacks.wandb.project=sewonm-icsl \
  --trainer.callbacks.wandb.name="${runname}" \
  --trainer.callbacks.wandb.tags="[pretraining, step0, 0802, repeated-data, dclm-train-only, ${scheduler}]" \
  --trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
  --trainer.callbacks.downstream_evaluator.eval_interval=null \
  --trainer.callbacks.downstream_evaluator.eval_on_finish=true \
  --trainer.callbacks.heldout_evaluator="{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, mix: null, mix_base_dir: ${DATA_ROOT}, subset_manifest: ${validation_manifest}, metadata: [{label: dclm-validation-0802}], sequence_length: 4096, work_dir: ${DATASET_CACHE}}, eval_interval: null, eval_on_finish: true, name: heldout}" \
  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
  --model.block.name=default \
  --model.block.sequence_mixer.qk_norm=null \
  --train_module.scheduler="${scheduler_config}" \
  --trainer.callbacks.checkpointer.fixed_steps="${checkpoint_steps}" \
  --init_seed="${init_seed}" \
  --data_loader.seed="${data_seed}" \
  --train_module.optim.weight_decay="${wd}" \
  --lr="${lr}"
