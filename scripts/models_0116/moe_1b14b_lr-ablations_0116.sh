# PARENT: "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123.sh"
# DESCRIPTION:
#     - Same as parent except cleaned up name, set learning rate to default
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

lr=4e-3
lb=1e-1

runname="moe_1b14b_lr-${lr}_lb-${lb}-ablations_0116"
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
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.no_checkpoints \
    --trainer.hard_stop='{value: 5000, unit: steps}'











