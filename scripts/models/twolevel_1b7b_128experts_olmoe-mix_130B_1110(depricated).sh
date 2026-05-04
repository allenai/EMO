# PARENT: N/A
# DESCRIPTION:
#     - implemented two-level MoE
#     - different from standard MoE (e.g moe_1b14b_128experts_olmoe-mix_130B_1117.sh) in the following:
#         1) do intra-document masking
#         2) Has a expert-pool which is selected for each document (here it's set to 32)
# STATUS: DEPRICATED
#     - replaced by "twolevelbatchlb_1b14b_stability_filter-true_zlossweight-1e-3_1115.sh" since performance was not good and training was unstable (loss was not consistenly decreasing)
##############################################################
document_expert_pool=32

runname="twolevel-${document_expert_pool}_1b7b_128experts_olmoe-mix_130B_1110"

# torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
#    $runname \
#		--save-folder="/root/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
#		--dataset.mix=arc-easy-train \
#		--work-dir="/root/ryanwang/dataset-cache" \
#		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#		--trainer.callbacks.wandb.enabled=true \
#		--trainer.callbacks.wandb.entity=ryanyxw \
#		--trainer.callbacks.wandb.project=olmoe-modular \
#		--trainer.callbacks.wandb.name="${runname}" \
#		--model.block.feed_forward_moe.num_experts=16 \
#		--model-type="two-level" \
#		--document-expert-pool=${document_expert_pool} \
#		--train_module.compile_model=false


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
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level" \
		--document-expert-pool=${document_expert_pool}








