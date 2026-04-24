# PARENT: "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121.sh"
# DESCRIPTION:
#     - the same but added on some new elements
# STATUS: USED
##############################################################
document_expert_pool=32
lr=4e-3
lb=1e-2

nodes=16
gpus=8
# calculate by taking nodes multiply by gpus multiply by 4 (since we have 4 as micro batch size)
lb_global_batch_size=$((nodes * gpus * 4))

num_shared_experts_pool=4
num_shared_experts=2 # 2 out of 4 will be shared experts. NOTE: cannot set to 1 because softmax gradients won't backprop
shared_exp_lb_loss=1e-2 # coefficient for the lb loss for shared experts, in reality is 1/16 (divide by number of layers)

runname="twolevelbatchlbreducedp${lb_global_batch_size}sharedexp${num_shared_experts_pool}c${num_shared_experts}-${document_expert_pool}_1b14b_lr-${lr}_lb-${lb}_sharelb-${shared_exp_lb_loss}_0214"


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
#  --global_batch_size=2 \
#  --model.block.feed_forward_moe.num_experts=16 \
#  --model-type="two-level_lb-batch_reduce-dp_sharedexppool" \
#  --document-expert-pool=10 \
#  --num_shared_experts=1 \
#  --num_shared_experts_pool=4 \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.name="moe" \
#  --model.block.sequence_mixer.qk_norm=null \
#  --lr=${lr} \
#  --model.block.feed_forward_moe.lb_loss_weight=${lb}


python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus $gpus \
  --nodes $nodes \
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
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexppool" \
		--document-expert-pool=${document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
    --num_shared_experts_pool=$num_shared_experts_pool \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--shared_exp_lb_loss=${shared_exp_lb_loss}
