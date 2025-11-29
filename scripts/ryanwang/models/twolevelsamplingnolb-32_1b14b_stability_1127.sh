# PARENT: "twolevelsamplingnolb_1b14b_stability_filter-true_zlossweight-1e-3_1116.sh
# DESCRIPTION:
#     - add prenorm and remove qknorm
# STATUS: USED
##############################################################
document_expert_pool=32
#document_expert_pool=12

runname="twolevelsamplingnolb-${document_expert_pool}_1b14b_stability_1127"

#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
#  $runname \
#  --save-folder="/root/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="/root/ryanwang/dataset-cache" \
#  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#  --trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
#  --model.block.feed_forward_moe.num_experts=16 \
#  --model-type="two-level_sampling_nolb" \
#  --document-expert-pool=${document_expert_pool} \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.feed_forward_moe.z_loss_weight=0.001 \
#  --model.block.feed_forward_moe.lb_loss_weight=0 \


python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus 8 \
  --nodes 8 \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
  --is_private_repo \
	--preemptible \
	--allow-dirty \
	--priority urgent \
	--env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
	-- src/scripts/train/olmoe-1B-7B_fsl.py \
    $runname \
		--save-folder="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.attention.backend=flash_2 \
		--model-type="two-level_sampling_nolb" \
		--document-expert-pool=${document_expert_pool} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.feed_forward_moe.lb_loss_weight=0 \
    --model.block.name="moe" \
		--model.block.attention.qk_norm=null








