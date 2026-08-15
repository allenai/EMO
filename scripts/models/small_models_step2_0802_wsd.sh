#!/usr/bin/env bash
# DESCRIPTION:
#     - Launches the matched 153M or 474M dense/standard-MoE models for the
#       0802 unique-versus-repeated WSD study.
#     - Dense and MoE counterparts have identical backbone dimensions and
#       active FFN width. Standard MoEs use 128 experts, top-8 activation,
#       and one always-active shared expert (seven experts are routed).
#     - Uses a 100M-token warmup at both scales: 95 steps with a 1 Mi-token
#       batch at 153M and 48 steps with a 2 Mi-token batch at 474M.
# USAGE:
#     ARCHITECTURE=dense MODEL_SIZE=153M REGIME=unique EPOCHS=1 LR=5e-4 \
#       MODE=beaker bash scripts/models/small_models_step2_0802_wsd.sh
#     ARCHITECTURE=stdmoe MODEL_SIZE=474M REGIME=repeated EPOCHS=1 LR=5e-4 \
#       MODE=beaker bash scripts/models/small_models_step2_0802_wsd.sh
#     Targets are 1, 2, 4, 8, 12, and 16B tokens. Each segment preserves the
#     pre-decay checkpoint for every intermediate integer-billion-token epoch,
#     so intermediate evaluations can be added later without retraining.
#     Later targets resume the preceding target's pre-decay checkpoint from the
#     same architecture, size, regime, and LR chain. START_FRESH=1 introduces
#     a new LR at a later target without LOAD_PATH and preserves epochs 1..N.
##############################################################
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/ai2-llm}"
BEAKER_PREEMPTIBLE="${BEAKER_PREEMPTIBLE:-0}"
BEAKER_AUTO_RESUME="${BEAKER_AUTO_RESUME:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

architecture="${ARCHITECTURE:?Set ARCHITECTURE to dense or stdmoe}"
model_size="${MODEL_SIZE:?Set MODEL_SIZE to 153M or 474M}"
regime="${REGIME:?Set REGIME to unique or repeated}"
epochs="${EPOCHS:?Set EPOCHS to one of 1, 2, 4, 8, 12, or 16}"
lr="${LR:?Set LR}"
wd="${WD:-0.033}"
lb="${LB:-0.1}"
start_fresh="${START_FRESH:-0}"
init_seed="${INIT_SEED:-12536}"
data_seed="${DATA_SEED:-0}"
run_suffix="${RUN_SUFFIX:-}"
validation_manifest="${VALIDATION_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_validation.json}"

case "${architecture}" in
	dense|stdmoe) ;;
	*) echo "ARCHITECTURE must be dense or stdmoe" >&2; exit 2 ;;
esac

case "${model_size}" in
	153M)
		model_slug="153m"
		moe_slug="153m720m"
		d_model=512
		n_layers=12
		n_heads=8
		dense_hidden_size=2048
		expert_hidden_size=256
		global_batch_tokens=1048576
		global_batch_sequences=256
		warmup_steps=95
		;;
	474M)
		model_slug="474m"
		moe_slug="474m3b5"
		d_model=1024
		n_layers=16
		n_heads=16
		dense_hidden_size=4096
		expert_hidden_size=512
		global_batch_tokens=2097152
		global_batch_sequences=512
		warmup_steps=48
		;;
	*) echo "MODEL_SIZE must be 153M or 474M" >&2; exit 2 ;;
esac

case "${regime}" in
	unique)
		subset_manifest="${SUBSET_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_unique_train_5b.json}"
		data_slug="unique_dclm5b"
		;;
	repeated)
		subset_manifest="${SUBSET_MANIFEST:-src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json}"
		data_slug="repeated_dclm1b"
		;;
	*) echo "REGIME must be unique or repeated" >&2; exit 2 ;;
esac

case "${epochs}" in
	1|2|4|8|12|16) max_tokens=$((epochs * 1000000000)) ;;
	*) echo "EPOCHS must be one of: 1, 2, 4, 8, 12, 16" >&2; exit 2 ;;
esac

case "${lr}" in
	1.25e-4|2.5e-4|5e-4|1e-3|2e-3|4e-3) ;;
	*) echo "LR must be one of: 1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3" >&2; exit 2 ;;
esac

if [[ "${start_fresh}" != "0" && "${start_fresh}" != "1" ]]; then
	echo "START_FRESH must be 0 or 1" >&2
	exit 2
fi
if [[ ! "${init_seed}" =~ ^[0-9]+$ || ! "${data_seed}" =~ ^[0-9]+$ ]]; then
	echo "INIT_SEED and DATA_SEED must be non-negative integers" >&2
	exit 2
fi

for manifest in "${subset_manifest}" "${validation_manifest}"; do
	if [[ ! -f "${manifest}" ]]; then
		echo "Required 0802 manifest does not exist: ${manifest}" >&2
		exit 2
	fi
done

# Return round(steps / 10), matching Python's ties-to-even rounding used by
# WSD when decay_fraction=0.1.
round_tenth() {
	local steps="$1"
	local quotient=$((steps / 10))
	local remainder=$((steps % 10))
	if ((remainder > 5 || (remainder == 5 && quotient % 2 == 1))); then
		quotient=$((quotient + 1))
	fi
	echo "${quotient}"
}

pre_decay_step() {
	local target_tokens="$1"
	local max_steps=$(((target_tokens + global_batch_tokens - 1) / global_batch_tokens))
	local decay_steps
	decay_steps="$(round_tenth "${max_steps}")"
	echo $((max_steps - decay_steps - 1))
}

load_args=()
dry_run_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then dry_run_args+=(--dry-run); fi
if [[ "${epochs}" -eq 1 || "${start_fresh}" == "1" ]]; then
	if [[ -n "${LOAD_PATH:-}" ]]; then
		echo "A fresh LR chain must not set LOAD_PATH" >&2
		exit 2
	fi
else
	: "${LOAD_PATH:?Set LOAD_PATH to the preceding pre-decay checkpoint from the same chain}"
	load_args+=(--load_path="${LOAD_PATH}" --load_trainer_state=true)
fi

if [[ "${start_fresh}" == "1" ]]; then
	checkpoint_start_epoch=1
else
	case "${epochs}" in
		1) checkpoint_start_epoch=1 ;;
		2) checkpoint_start_epoch=2 ;;
		4) checkpoint_start_epoch=3 ;;
		8) checkpoint_start_epoch=5 ;;
		12) checkpoint_start_epoch=9 ;;
		16) checkpoint_start_epoch=13 ;;
	esac
fi

checkpoint_steps="["
for ((checkpoint_epoch = checkpoint_start_epoch; checkpoint_epoch <= epochs; checkpoint_epoch++)); do
	if [[ "${checkpoint_steps}" != "[" ]]; then checkpoint_steps+=","; fi
	checkpoint_steps+="$(pre_decay_step "$((checkpoint_epoch * 1000000000))")"
done
checkpoint_steps+="]"

downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
scheduler_config="{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: ${warmup_steps}, decay_fraction: 0.1}"

model_args=()
if [[ "${architecture}" == "dense" ]]; then
	training_script="src/scripts/train/olmo2-1B.py"
	run_prefix="dense_${model_slug}"
	architecture_tags="dense-${model_slug}"
	# The pinned Beaker revision already has the 153M factory. Express 474M
	# as config overrides on the supported 1B factory so jobs do not depend on
	# uncommitted local parser changes.
	if [[ "${model_size}" == "153M" ]]; then
		model_args+=(--model-size=153M)
	else
		model_args+=(
			--model-size=1B
			--model.d_model="${d_model}"
			--model.n_layers="${n_layers}"
			--model.block.sequence_mixer.n_heads="${n_heads}"
			--model.block.feed_forward.hidden_size="${dense_hidden_size}"
		)
	fi
	model_args+=(
		--data_loader.global_batch_size="${global_batch_tokens}"
		--model.block.name=default
		--model.block.sequence_mixer.qk_norm=null
	)
else
	training_script="src/scripts/train/olmoe-1B-7B_fsl.py"
	run_prefix="stdmoe_${moe_slug}"
	architecture_tags="standard-moe, moe-${moe_slug}"
	model_args+=(
		--model.d_model="${d_model}"
		--model.n_layers="${n_layers}"
		--model.block.sequence_mixer.n_heads="${n_heads}"
		--model-type=moe_lbreducedp_sharedexp
		--num_shared_experts=1
		--model.block.feed_forward_moe.num_experts=128
		--model.block.feed_forward_moe.hidden_size="${expert_hidden_size}"
		--model.block.feed_forward_moe.lb_loss_weight="${lb}"
		--model.block.name=moe
		--model.block.sequence_mixer.qk_norm=null
		--global_batch_size="${global_batch_sequences}"
	)
fi

runname="${run_prefix}_step2_0802_${data_slug}_wsd_e${epochs}_lr${lr}_wd${wd}_warmup${warmup_steps}${run_suffix}"

launch "${training_script}" "${runname}" \
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
	--trainer.callbacks.wandb.tags="[pretraining, step2, 0802, ${architecture_tags}, ${regime}-data, dclm-train-only, document-disjoint, uniform-document-sample, wsd]" \
	--trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
	--trainer.callbacks.downstream_evaluator.eval_interval=null \
	--trainer.callbacks.downstream_evaluator.eval_on_finish=true \
	--trainer.callbacks.heldout_evaluator="{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, mix: null, mix_base_dir: ${DATA_ROOT}, subset_manifest: ${validation_manifest}, metadata: [{label: dclm-validation-0802}], sequence_length: 4096, work_dir: ${DATASET_CACHE}}, eval_interval: null, eval_on_finish: true, name: heldout}" \
	--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
	--train_module.scheduler="${scheduler_config}" \
	--trainer.callbacks.checkpointer.fixed_steps="${checkpoint_steps}" \
	"${model_args[@]}" \
	"${load_args[@]}" \
	--init_seed="${init_seed}" \
	--data_loader.seed="${data_seed}" \
	--train_module.optim.weight_decay="${wd}" \
	--lr="${lr}"
