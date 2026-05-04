# PARENT: "moe_1b4b_32experts_1224.sh"
# DESCRIPTION:
#     - Same as parent except increased learning rate
# STATUS: nan lossed
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

lr=4e-3
runname="moe_1b4b_lr-${lr}_0204"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining]' \
		--model-type="moe" \
		--model.block.feed_forward_moe.num_experts=32 \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr}











