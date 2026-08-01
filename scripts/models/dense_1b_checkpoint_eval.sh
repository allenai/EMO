#!/usr/bin/env bash
# Evaluate an exact dense-1B checkpoint on the shared nine-task suite and C4.
# This loads model weights only, evaluates before training, and cancels before
# the first optimizer step.
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
BEAKER_AUTO_RESUME="${BEAKER_AUTO_RESUME:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

epochs="${EPOCHS:?Set EPOCHS to an integer from 1 through 5}"
checkpoint_path="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the exact terminal checkpoint}"
runname="${RUN_NAME:?Set a unique RUN_NAME for this evaluation}"
lr="${LR:?Set LR to the checkpoint peak learning rate}"
wd="${WD:-0.033}"
dataset_mix="${DATASET_MIX:-dclm-full}"
downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
wandb_tags="${WANDB_TAGS:-[checkpoint-eval, downstream-nine]}"

case "${epochs}" in
	1) checkpoint_step=239 ;;
	2) checkpoint_step=477 ;;
	3) checkpoint_step=716 ;;
	4) checkpoint_step=954 ;;
	5) checkpoint_step=1193 ;;
	*) echo "EPOCHS must be an integer from 1 through 5" >&2; exit 2 ;;
esac

if [[ "${checkpoint_path}" != */step"${checkpoint_step}" ]]; then
	echo "CHECKPOINT_PATH must end in /step${checkpoint_step} for EPOCHS=${epochs}" >&2
	exit 2
fi

launch src/scripts/train/olmo2-1B.py "${runname}" \
		--save-folder="${MODELS_DIR}/${runname}" \
		--dataset.mix="${dataset_mix}" \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 6000000000, unit: tokens}' \
		--trainer.callbacks.checkpointer.enabled=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="${wandb_tags}" \
		--trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
		--trainer.callbacks.downstream_evaluator.eval_interval=null \
		--trainer.callbacks.downstream_evaluator.eval_on_finish=false \
		--trainer.callbacks.downstream_evaluator.eval_on_startup=true \
		--trainer.callbacks.downstream_evaluator.cancel_after_first_eval=true \
		--trainer.callbacks.heldout_evaluator='{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyPaddedFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, paths: [/weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy], metadata: [{label: c4_en-validation}], sequence_length: 4096, work_dir: /weka/oe-training-default/sewonm/dataset-cache}, eval_interval: null, eval_on_startup: true, eval_on_finish: false, name: heldout}' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name=default \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.optim.weight_decay="${wd}" \
		--load_path="${checkpoint_path}" \
		--load_trainer_state=false \
		--lr="${lr}"
