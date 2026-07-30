# PARENT: "dense_1b_1b.sh"
# DESCRIPTION:
#     - Uses a warmup-stable-decay learning-rate schedule
# STATUS: EXPERIMENTAL
##############################################################
DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/sewonm}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"


lr=4e-3

runname="dense_1b_1b_wsd"
launch src/scripts/train/olmo2-1B.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=codelion-dclm-baseline-1B \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 1_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining, wsd]' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="default" \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler='{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 20, decay_fraction: 0.1}' \
		--lr=${lr}
