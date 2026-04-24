# PARENT: "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123.sh"
# DESCRIPTION:
#     - First implementation of mutual-info model. Use prenorm + noqknorm, no intra-document masking
# STATUS: DEPRICATED
#     = The loss does not converge well, unique experts increases too fast, grad norm spikes very high
#     - Add z-loss to reduce grad norm
##############################################################

runname="mutualinfo_1b14b_cond-1_uncond-1_1205"

#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
#  $runname \
#  --save-folder="/root/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="/root/ryanwang/dataset-cache" \
#  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#  --trainer.callbacks.wandb.enabled=false \
#  --trainer.callbacks.wandb.entity=ryanyxw \
#  --trainer.callbacks.wandb.project=olmoe-modular \
#  --trainer.callbacks.wandb.name="${runname}" \
#  --model.block.feed_forward_moe.num_experts=16 \
#  --model-type="mutual-info" \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.name="moe" \
#	--model.block.sequence_mixer.qk_norm=null \
#	--model.block.feed_forward_moe.z_loss_weight=null \
#	--model.block.feed_forward_moe.lb_loss_weight=null \
#	--expert_cond_token_entropy_bias=1 \
#	--expert_uncond_entropy_bias=1



python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus 8 \
  --nodes 8 \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
  --beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
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
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining]' \
		--model-type="mutual-info" \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
	  --model.block.feed_forward_moe.z_loss_weight=null \
	  --model.block.feed_forward_moe.lb_loss_weight=null \
	  --expert_cond_token_entropy_bias=1 \
	  --expert_uncond_entropy_bias=1








