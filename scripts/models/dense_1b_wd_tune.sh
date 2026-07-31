#!/usr/bin/env bash
# DESCRIPTION:
#     - Runs one fresh WSD target for the dense-1B weight-decay study.
#     - Every (epoch, LR, WD) tuple trains from scratch so optimizer history
#       from a different LR or weight decay cannot confound comparisons.
#     - Weight decay is applied as a CLI override; the Python config is unchanged.
# USAGE:
#     EPOCHS=3 LR=5e-4 WD=0.1 MODE=beaker \
#       bash scripts/models/dense_1b_wd_tune.sh
# STATUS: EXPERIMENTAL
##############################################################
set -eo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/sewonm}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

epochs="${EPOCHS:?Set EPOCHS to an integer from 1 through 5}"
lr="${LR:?Set LR}"
wd="${WD:?Set WD}"

case "${epochs}" in
	1) max_tokens=1000000000 ;;
	2) max_tokens=2000000000 ;;
	3) max_tokens=3000000000 ;;
	4) max_tokens=4000000000 ;;
	5) max_tokens=5000000000 ;;
	*)
		echo "EPOCHS must be an integer from 1 through 5" >&2
		exit 2
		;;
esac

# One epoch is 239 optimizer steps, so round(10%) is 24 steps.
warmup_steps=24
downstream_tasks="${DOWNSTREAM_TASKS:-[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]}"
dry_run_args=()

if [[ "${DRY_RUN:-0}" == "1" ]]; then
	dry_run_args+=(--dry-run)
fi

scheduler_config="{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: ${warmup_steps}, decay_fraction: 0.1}"
runname="dense_1b_wd_tune_wsd_e${epochs}_lr${lr}_wd${wd}_warmup${warmup_steps}"

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
		--trainer.callbacks.wandb.tags="[pretraining, wd-tune, wsd, warmup24]" \
		--trainer.callbacks.downstream_evaluator.tasks="${downstream_tasks}" \
		--trainer.callbacks.downstream_evaluator.eval_interval=null \
		--trainer.callbacks.downstream_evaluator.eval_on_finish=true \
		--trainer.callbacks.heldout_evaluator='{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyPaddedFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, paths: [/weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy], metadata: [{label: c4_en-validation}], sequence_length: 4096, work_dir: /weka/oe-training-default/sewonm/dataset-cache}, eval_interval: null, eval_on_finish: true, name: heldout}' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name=default \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler="${scheduler_config}" \
		--train_module.optim.weight_decay="${wd}" \
		--lr="${lr}"
