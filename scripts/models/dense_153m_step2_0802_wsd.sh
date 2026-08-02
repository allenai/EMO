#!/usr/bin/env bash
# DESCRIPTION:
#     - Runs one WSD endpoint for the explicit dense 153M configuration:
#       current dense-1B architecture with d_model=512, 12 layers, 8 heads,
#       and a 2,048-wide FFN (153,104,896 parameters with Dolma2 vocabulary).
#     - At epoch 1 both regime labels consume the exact same repeated 1B corpus.
#       This makes the diagnostic matched by construction; a future unique-data
#       continuation requires a separately verified nested 5B ordering.
#     - A 1 Mi-token global batch yields enough optimizer updates for this
#       model scale while retaining the same 100M-token warmup budget.
# USAGE:
#     REGIME=unique EPOCHS=1 LR=5e-4 INIT_SEED=12536 DATA_SEED=0 \
#       RUN_SUFFIX=_seed0_a MODE=beaker \
#       bash scripts/models/dense_153m_step2_0802_wsd.sh
##############################################################
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
BEAKER_AUTO_RESUME="${BEAKER_AUTO_RESUME:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

regime="${REGIME:?Set REGIME to unique or repeated}"
epochs="${EPOCHS:?Set EPOCHS to an integer from 1 through 5}"
lr="${LR:?Set LR}"
wd="${WD:-0.033}"
init_seed="${INIT_SEED:-12536}"
data_seed="${DATA_SEED:-0}"
run_suffix="${RUN_SUFFIX:?Set a unique RUN_SUFFIX for this replicate}"
validation_manifest="${VALIDATION_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_validation.json}"
global_batch_size="${GLOBAL_BATCH_SIZE:-1048576}"
warmup_steps="${WARMUP_STEPS:-95}"

case "${regime}" in
	unique|repeated) subset_manifest="${SUBSET_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json}" ;;
	*) echo "REGIME must be unique or repeated" >&2; exit 2 ;;
esac

case "${lr}" in
	5e-4|1e-3|2e-3|4e-3) ;;
	*) echo "LR must be one of: 5e-4, 1e-3, 2e-3, 4e-3" >&2; exit 2 ;;
esac

case "${epochs}" in
	1) max_tokens=1000000000; stable_step=858 ;;
	*) echo "Only matched epoch 1 is authorized; later unique endpoints require a verified nested 5B ordering" >&2; exit 2 ;;
esac

if [[ ! "${init_seed}" =~ ^[0-9]+$ || ! "${data_seed}" =~ ^[0-9]+$ ]]; then
	echo "INIT_SEED and DATA_SEED must be non-negative integers" >&2
	exit 2
fi
if [[ "${global_batch_size}" != "1048576" ]]; then
	echo "GLOBAL_BATCH_SIZE must remain 1048576 because fixed checkpoint steps depend on it" >&2
	exit 2
fi

for manifest in "${subset_manifest}" "${validation_manifest}"; do
	if [[ ! -f "${manifest}" ]]; then
		echo "Required 0802 manifest does not exist: ${manifest}" >&2
		exit 2
	fi
done

load_args=()
dry_run_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then dry_run_args+=(--dry-run); fi
if [[ "${epochs}" -eq 1 ]]; then
	if [[ -n "${LOAD_PATH:-}" ]]; then
		echo "Epoch 1 starts fresh; do not set LOAD_PATH" >&2
		exit 2
	fi
else
	: "${LOAD_PATH:?Set LOAD_PATH to the matching replicate/LR pre-decay checkpoint}"
	load_args+=(--load_path="${LOAD_PATH}" --load_trainer_state=true)
fi

downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
scheduler_config="{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: ${warmup_steps}, decay_fraction: 0.1}"
runname="dense_153m_step2_0802_${regime}_wsd_e${epochs}_lr${lr}_wd${wd}_warmup${warmup_steps}${run_suffix}"

launch src/scripts/train/olmo2-1B.py "${runname}" \
		"${dry_run_args[@]}" \
		--model-size=153M \
		--save-folder="${MODELS_DIR}/${runname}" \
		--dataset.mix=null \
		--dataset.subset_manifest="${subset_manifest}" \
		--dataset.mix_base_dir="${DATA_ROOT}" \
		--work-dir="${DATASET_CACHE}" \
		--data_loader.global_batch_size="${global_batch_size}" \
		--trainer.max_duration="{value: ${max_tokens}, unit: tokens}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, step2, 0802, dense-153m, ${regime}-data, dclm-train-only, seed-replicate, wsd]" \
		--trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
		--trainer.callbacks.downstream_evaluator.eval_interval=null \
		--trainer.callbacks.downstream_evaluator.eval_on_finish=true \
		--trainer.callbacks.heldout_evaluator="{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, mix: null, mix_base_dir: ${DATA_ROOT}, subset_manifest: ${validation_manifest}, metadata: [{label: dclm-validation-0802}], sequence_length: 4096, work_dir: ${DATASET_CACHE}}, eval_interval: null, eval_on_finish: true, name: heldout}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name=default \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler="${scheduler_config}" \
		--trainer.callbacks.checkpointer.fixed_steps="[${stable_step}]" \
		"${load_args[@]}" \
		--init_seed="${init_seed}" \
		--data_loader.seed="${data_seed}" \
		--train_module.optim.weight_decay="${wd}" \
		--lr="${lr}"
