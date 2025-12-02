# PARENT: "moe_1b7b_olmoe-mix.sh"
# DESCRIPTION:
#     - Same as parent except trained on 300B tokens instead of 130B tokens
# STATUS: DEPRICATED
#     - Decided to (1) stick with 130B and (2) use 128 experts instead of 64. See "moe_1b7b_128experts_olmoe-mix_130B_1103.sh"
##############################################################

runname="moe_1b7b_olmoe-mix_300B_1030"
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
		--trainer.max_duration='{value: 300_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \








