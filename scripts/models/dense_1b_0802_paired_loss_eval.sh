#!/usr/bin/env bash
# Emit per-document validation-loss sufficient statistics for one exact 0802 endpoint.
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_GPUS="${BEAKER_GPUS:-1}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
BEAKER_AUTO_RESUME="${BEAKER_AUTO_RESUME:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

checkpoint_path="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the exact decayed endpoint checkpoint}"
runname="${RUN_NAME:?Set a unique RUN_NAME}"
lr="${LR:?Set LR}"
wd="${WD:-0.033}"
validation_manifest="${VALIDATION_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_validation.json}"
document_metadata_path="${DOCUMENT_METADATA_PATH:?Set DOCUMENT_METADATA_PATH to dclm_0802_validation_uint32.csv.gz}"
paired_stats_output="${PAIRED_STATS_OUTPUT:?Set a fresh PAIRED_STATS_OUTPUT .npz path}"

if [[ ! -f "${validation_manifest}" ]]; then
	echo "Validation manifest does not exist: ${validation_manifest}" >&2
	exit 2
fi
if [[ "${paired_stats_output}" != *.npz ]]; then
	echo "PAIRED_STATS_OUTPUT must end in .npz" >&2
	exit 2
fi

launch src/scripts/train/olmo2-1B.py "${runname}" \
		--save-folder="${MODELS_DIR}/${runname}" \
		--dataset.mix=null \
		--dataset.subset_manifest="${validation_manifest}" \
		--dataset.mix_base_dir="${DATA_ROOT}" \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 6000000000, unit: tokens}' \
		--trainer.callbacks.checkpointer.enabled=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[checkpoint-eval, step2, 0802, paired-document-loss]' \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.callbacks.heldout_evaluator="{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, mix: null, mix_base_dir: ${DATA_ROOT}, subset_manifest: ${validation_manifest}, metadata: [{label: dclm-validation-0802}], sequence_length: 4096, work_dir: ${DATASET_CACHE}}, eval_interval: null, eval_on_startup: true, eval_on_finish: false, cancel_after_first_eval: true, deterministic: true, paired_document_metadata_path: ${document_metadata_path}, paired_stats_output_path: ${paired_stats_output}, name: heldout}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name=default \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.rank_microbatch_size=4096 \
		--train_module.optim.weight_decay="${wd}" \
		--load_path="${checkpoint_path}" \
		--load_trainer_state=false \
		--lr="${lr}"
