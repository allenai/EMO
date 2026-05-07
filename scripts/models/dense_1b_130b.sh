# PARENT: "dense_1b_olmoe-mix_1019.sh"
# DESCRIPTION:
#     - Different from parent by removing qknorm and using prenorm
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"


lr=4e-3

runname="dense_1b_130b"
launch src/scripts/train/olmo2-1B.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining]' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="default" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr}










