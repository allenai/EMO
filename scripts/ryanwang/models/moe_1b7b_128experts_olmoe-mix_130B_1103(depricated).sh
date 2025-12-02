# PARENT: "moe_1b7b_olmoe-mix.sh"
# DESCRIPTION:
#     - same as parent but with 128 experts (total 14B params) trained on 130B tokens
#     - typo in name (should be 1b14b instead of 1b7b),
# STATUS: DEPRICATED
#     - replaced by moe_1b14b_128experts_olmoe-mix_130B_1117.sh for replication of in-loop evals and updated dataset filtering
##############################################################

runname="moe_1b7b_128experts_olmoe-mix_130B_1103"
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








