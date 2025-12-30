# PARENT: "twolevelbatchlb-32_1b14b_stability_prenorm_1120.sh"
# DESCRIPTION:
#     - removed qk-norm from the model compared to parent.
# STATUS: USED
##############################################################
document_expert_pool=32
#lb_loss_weight=0.01
lb_loss_weight=0
uncond_lb_bias=1e-1
uncond_bias=0

runname="twolevelpbatchlb-${document_expert_pool}_1b14b_uncond-${uncond_bias}_uncondlb-${uncond_lb_bias}_lbloss-${lb_loss_weight}_1230"

#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
#  $runname \
#  --save-folder="/root/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="/root/ryanwang/dataset-cache" \
#  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#  --trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
#  --model.block.feed_forward_moe.num_experts=16 \
#  --model-type="two-level" \
#  --document-expert-pool=${document_expert_pool} \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.feed_forward_moe.z_loss_weight=0.004 \


python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus 8 \
  --nodes 16 \
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
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}, tags: [pretraining]}" \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.attention.backend=flash_2 \
		--model-type="two-level_p_lb-batch" \
		--document-expert-pool=${document_expert_pool} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.attention.qk_norm=null \
	  --model.block.feed_forward_moe.lb_loss_weight=${lb_loss_weight} \
	  --expert_uncond_entropy_bias=${uncond_bias} \
	  --expert_uncond_lb_prob_bias=${uncond_lb_bias} \
    --trainer.hard_stop='{value: 5000, unit: steps}'








