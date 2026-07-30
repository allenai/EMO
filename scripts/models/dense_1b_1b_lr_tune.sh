# DESCRIPTION:
#     - Runs one point in the matched cosine/WSD learning-rate sweep
# USAGE:
#     SCHEDULER=cosine LR=2e-3 MODE=beaker bash scripts/models/dense_1b_1b_lr_tune.sh
#     SCHEDULER=wsd LR=2e-3 MODE=beaker bash scripts/models/dense_1b_1b_lr_tune.sh
# STATUS: EXPERIMENTAL
##############################################################
DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/sewonm}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"


lr="${LR:?Set LR (for example, LR=2e-3)}"
scheduler="${SCHEDULER:?Set SCHEDULER to cosine or wsd}"

case "${scheduler}" in
	cosine)
		scheduler_config='{_CLASS_: olmo_core.optim.scheduler.CosWithWarmup, units: steps, warmup: 20, alpha_f: 0.1}'
		;;
	wsd)
		scheduler_config='{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 20, decay_fraction: 0.1}'
		;;
	*)
		echo "SCHEDULER must be 'cosine' or 'wsd'" >&2
		exit 2
		;;
esac

runname="dense_1b_1b_${scheduler}_lr${lr}"
launch src/scripts/train/olmo2-1B.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=codelion-dclm-baseline-1B \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 1_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, lr-sweep, ${scheduler}]" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="default" \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler="${scheduler_config}" \
		--lr=${lr}
