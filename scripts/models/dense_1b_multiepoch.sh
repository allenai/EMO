#!/usr/bin/env bash
# DESCRIPTION:
#     - Runs one target in the dense-1B cosine/WSD multi-epoch study.
#     - Cosine targets always train from scratch.
#     - WSD targets after epoch 1 resume from the preceding target's last stable
#       (pre-decay) checkpoint with optimizer, trainer, and data-loader state.
# USAGE:
#     SCHEDULER=cosine EPOCHS=3 LR=2.5e-4 MODE=beaker \
#       bash scripts/models/dense_1b_multiepoch.sh
#     SCHEDULER=cosine COSINE_ALPHA_F=0 EPOCHS=1 LR=2.5e-4 MODE=beaker \
#       bash scripts/models/dense_1b_multiepoch.sh
#     SCHEDULER=wsd EPOCHS=2 LR=5e-4 \
#       LOAD_PATH=/weka/.../dense_1b_multiepoch_wsd_e1_lr5e-4_warmup24/step214 \
#       MODE=beaker bash scripts/models/dense_1b_multiepoch.sh
# STATUS: EXPERIMENTAL
##############################################################
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/sewonm}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

scheduler="${SCHEDULER:?Set SCHEDULER to cosine or wsd}"
epochs="${EPOCHS:?Set EPOCHS to an integer from 1 through 10}"
lr="${LR:?Set LR}"

case "${epochs}" in
	1) max_tokens=1000000000; stable_step=214; epoch_checkpoint_steps='[239]' ;;
	2) max_tokens=2000000000; stable_step=428; epoch_checkpoint_steps='[239, 477]' ;;
	3) max_tokens=3000000000; stable_step=643; epoch_checkpoint_steps='[239, 477, 716]' ;;
	4) max_tokens=4000000000; stable_step=858; epoch_checkpoint_steps='[239, 477, 716, 954]' ;;
	5) max_tokens=5000000000; stable_step=1073; epoch_checkpoint_steps='[239, 477, 716, 954, 1193]' ;;
	6) max_tokens=6000000000; stable_step=1287; epoch_checkpoint_steps='[239, 477, 716, 954, 1193, 1431]' ;;
	7) max_tokens=7000000000; stable_step=1501; epoch_checkpoint_steps='[239, 477, 716, 954, 1193, 1431, 1670]' ;;
	8) max_tokens=8000000000; stable_step=1716; epoch_checkpoint_steps='[239, 477, 716, 954, 1193, 1431, 1670, 1908]' ;;
	9) max_tokens=9000000000; stable_step=1930; epoch_checkpoint_steps='[239, 477, 716, 954, 1193, 1431, 1670, 1908, 2147]' ;;
	10) max_tokens=10000000000; stable_step=2146; epoch_checkpoint_steps='[239, 477, 716, 954, 1193, 1431, 1670, 1908, 2147, 2385]' ;;
	*)
		echo "EPOCHS must be an integer from 1 through 10" >&2
		exit 2
		;;
esac

# One epoch is 239 optimizer steps, so round(10%) is 24 steps.
warmup_steps=24
# Future runs evaluate the OLMES core nine-task suite on completion. This remains
# a CLI override, and can be changed per submission with DOWNSTREAM_TASKS.
downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
load_args=()
checkpoint_args=()
dry_run_args=()
scheduler_suffix=""
scheduler_tag="${scheduler}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
	dry_run_args+=(--dry-run)
fi

case "${scheduler}" in
	cosine)
		if [[ -n "${LOAD_PATH:-}" ]]; then
			echo "Cosine targets must train from scratch; do not set LOAD_PATH" >&2
			exit 2
		fi
		cosine_alpha_f="${COSINE_ALPHA_F:-0.1}"
		if [[ "${cosine_alpha_f}" != "0.1" ]]; then
			alpha_name="${cosine_alpha_f//./p}"
			scheduler_suffix="_alpha${alpha_name}"
		fi
		scheduler_tag="cosine-alpha${cosine_alpha_f}"
		scheduler_config="{_CLASS_: olmo_core.optim.scheduler.CosWithWarmup, units: steps, warmup: ${warmup_steps}, alpha_f: ${cosine_alpha_f}}"
		# Preserve every completed epoch so downstream evaluation is recoverable.
		checkpoint_args+=(
			"--trainer.callbacks.checkpointer.fixed_steps=${epoch_checkpoint_steps}"
		)
		;;
	wsd)
		scheduler_config="{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: ${warmup_steps}, decay_fraction: 0.1}"
		checkpoint_args+=(
			"--trainer.callbacks.checkpointer.fixed_steps=[${stable_step}]"
		)
		if [[ "${epochs}" -eq 1 ]]; then
			if [[ -n "${LOAD_PATH:-}" ]]; then
				echo "WSD epoch 1 is the fresh phase-two baseline; do not set LOAD_PATH" >&2
				exit 2
			fi
		else
			: "${LOAD_PATH:?Set LOAD_PATH to the preceding WSD target stable checkpoint}"
			load_args+=(
				"--load_path=${LOAD_PATH}"
				"--load_trainer_state=true"
			)
		fi
		;;
	*)
		echo "SCHEDULER must be 'cosine' or 'wsd'" >&2
		exit 2
		;;
esac

runname="dense_1b_multiepoch_${scheduler}_e${epochs}_lr${lr}_warmup${warmup_steps}${scheduler_suffix}"
launch src/scripts/train/olmo2-1B.py "${runname}" \
		"${dry_run_args[@]}" \
		--save-folder="${MODELS_DIR}/${runname}" \
		--dataset.mix=codelion-dclm-baseline-1B \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${max_tokens}, unit: tokens}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, multiepoch, ${scheduler_tag}, warmup24]" \
		--trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
		--trainer.callbacks.downstream_evaluator.eval_interval=null \
		--trainer.callbacks.downstream_evaluator.eval_on_finish=true \
		--trainer.callbacks.heldout_evaluator='{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyPaddedFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, paths: [/weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy], metadata: [{label: c4_en-validation}], sequence_length: 4096, work_dir: /weka/oe-training-default/sewonm/dataset-cache}, eval_interval: null, eval_on_finish: true, name: heldout}' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name=default \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler="${scheduler_config}" \
		"${checkpoint_args[@]}" \
		"${load_args[@]}" \
		--lr="${lr}"
