#!/usr/bin/env bash
# Run one dense-1B Step-1 WSD weight-decay endpoint on verified 0802 Pool A.
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
BEAKER_AUTO_RESUME="${BEAKER_AUTO_RESUME:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

epochs="${EPOCHS:?Set EPOCHS to one of 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, or 24}"
lr="${LR:-5e-4}"
wd="${WD:?Set WD}"
subset_manifest="${SUBSET_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json}"
validation_manifest="${VALIDATION_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_validation.json}"
init_seed="${INIT_SEED:-12536}"
data_seed="${DATA_SEED:-0}"
run_suffix="${RUN_SUFFIX:-}"

case "${lr}" in 2.5e-4|5e-4|1e-3|2e-3|4e-3) ;; *) echo "Unsupported LR: ${lr}" >&2; exit 2 ;; esac
case "${epochs}" in
  1) max_tokens=1000000000; checkpoint_steps='[214]' ;;
  2) max_tokens=2000000000; checkpoint_steps='[428]' ;;
  3) max_tokens=3000000000; checkpoint_steps='[643]' ;;
  4) max_tokens=4000000000; checkpoint_steps='[858]' ;;
  5) max_tokens=5000000000; checkpoint_steps='[1073]' ;;
  6) max_tokens=6000000000; checkpoint_steps='[1287]' ;;
  8) max_tokens=8000000000; checkpoint_steps='[1502,1716]' ;;
  10) max_tokens=10000000000; checkpoint_steps='[1931,2145]' ;;
  12) max_tokens=12000000000; checkpoint_steps='[2360,2574]' ;;
  16) max_tokens=16000000000; checkpoint_steps='[2789,3003,3218,3432]' ;;
  20) max_tokens=20000000000; checkpoint_steps='[3647,3861,4076,4290]' ;;
  24) max_tokens=24000000000; checkpoint_steps='[4505,4719,4934,5148]' ;;
  *) echo "EPOCHS must be one of 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, or 24" >&2; exit 2 ;;
esac
for manifest in "${subset_manifest}" "${validation_manifest}"; do
  [[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 2; }
done
load_args=()
if [[ "${epochs}" -eq 1 ]]; then
  [[ -z "${LOAD_PATH:-}" ]] || { echo "Epoch 1 must start fresh" >&2; exit 2; }
else
  : "${LOAD_PATH:?Set LOAD_PATH to the matching preceding pre-decay checkpoint}"
  load_args+=(--load_path="${LOAD_PATH}" --load_trainer_state=true)
fi

warmup_steps=24
downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
scheduler_config="{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: ${warmup_steps}, decay_fraction: 0.1}"
runname="dense_1b_step1_0802_repeated_dclm1b_wsd_e${epochs}_lr${lr}_wd${wd}_warmup${warmup_steps}${run_suffix}"

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
  --trainer.callbacks.wandb.tags="[pretraining, step1, 0802, repeated-data, dclm-train-only, wd-tune, wsd]" \
  --trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
  --trainer.callbacks.downstream_evaluator.eval_interval=null \
  --trainer.callbacks.downstream_evaluator.eval_on_finish=true \
  --trainer.callbacks.heldout_evaluator="{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, mix: null, mix_base_dir: ${DATA_ROOT}, subset_manifest: ${validation_manifest}, metadata: [{label: dclm-validation-0802}], sequence_length: 4096, work_dir: ${DATASET_CACHE}}, eval_interval: null, eval_on_finish: true, name: heldout}" \
  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
  --model.block.name=default \
  --model.block.sequence_mixer.qk_norm=null \
  --train_module.scheduler="${scheduler_config}" \
  --trainer.callbacks.checkpointer.fixed_steps="${checkpoint_steps}" \
  "${load_args[@]}" \
  --init_seed="${init_seed}" \
  --data_loader.seed="${data_seed}" \
  --train_module.optim.weight_decay="${wd}" \
  --lr="${lr}"
