#!/usr/bin/env bash
# DESCRIPTION:
#     - Runs one target-specific WSD endpoint by repeatedly cycling the fixed
#       DCLM subset recorded in dclm_full_1b.json.
#     - Subset membership comes from explicit instance ranges, never a seed.
#     - Later targets resume the preceding stable checkpoint from the same LR chain.
# USAGE:
#     EPOCHS=1 LR=5e-4 MODE=beaker \
#       bash scripts/models/dense_1b_step2_repeated_wsd.sh
##############################################################
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

epochs="${EPOCHS:?Set EPOCHS to an integer from 1 through 5}"
lr="${LR:?Set LR}"
wd="${WD:-0.033}"
subset_manifest="${SUBSET_MANIFEST:-src/olmo_core/data/subsets/dclm_full_1b.json}"

case "${epochs}" in
	1) max_tokens=1000000000; stable_step=214 ;;
	2) max_tokens=2000000000; stable_step=428 ;;
	3) max_tokens=3000000000; stable_step=643 ;;
	4) max_tokens=4000000000; stable_step=858 ;;
	5) max_tokens=5000000000; stable_step=1073 ;;
	*) echo "EPOCHS must be an integer from 1 through 5" >&2; exit 2 ;;
esac

if [[ ! -f "${subset_manifest}" ]]; then
	echo "Subset manifest does not exist: ${subset_manifest}" >&2
	exit 2
fi

warmup_steps=24
downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
load_args=()
dry_run_args=()

if [[ "${DRY_RUN:-0}" == "1" ]]; then
	dry_run_args+=(--dry-run)
fi

if [[ "${epochs}" -eq 1 ]]; then
	if [[ -n "${LOAD_PATH:-}" ]]; then
		echo "Epoch 1 starts a fresh LR chain; do not set LOAD_PATH" >&2
		exit 2
	fi
else
	: "${LOAD_PATH:?Set LOAD_PATH to the preceding stable checkpoint from the same LR chain}"
	load_args+=(--load_path="${LOAD_PATH}" --load_trainer_state=true)
fi

scheduler_config="{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: ${warmup_steps}, decay_fraction: 0.1}"
run_suffix="${RUN_SUFFIX:-}"
runname="dense_1b_step2_repeated_dclm1b_wsd_e${epochs}_lr${lr}_wd${wd}_warmup${warmup_steps}${run_suffix}"

launch src/scripts/train/olmo2-1B.py "${runname}" \
		"${dry_run_args[@]}" \
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
		--trainer.callbacks.wandb.tags="[pretraining, step2, repeated-data, dclm-subset-1b, wsd, warmup24]" \
		--trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
		--trainer.callbacks.downstream_evaluator.eval_interval=null \
		--trainer.callbacks.downstream_evaluator.eval_on_finish=true \
		--trainer.callbacks.heldout_evaluator='{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyPaddedFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, paths: [/weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy], metadata: [{label: c4_en-validation}], sequence_length: 4096, work_dir: /weka/oe-training-default/sewonm/dataset-cache}, eval_interval: null, eval_on_finish: true, name: heldout}' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name=default \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler="${scheduler_config}" \
		--trainer.callbacks.checkpointer.fixed_steps="[${stable_step}]" \
		"${load_args[@]}" \
		--train_module.optim.weight_decay="${wd}" \
		--lr="${lr}"
