# PARENT: N/A
# DESCRIPTION:
#     - moe model to replicate OLMoE 1B-7B (64 total experts, 8 active)
# STATUS: DEPRICATED
#     - decided to use 128 experts instead of 64. See "moe_1b7b_128experts_olmoe-mix_130B_1103.sh"
##############################################################

runname="moe_1b7b_olmoe-mix"
python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus 8 \
  --nodes 8 \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--preemptible \
	--allow-dirty \
	--priority urgent \
	--env-secret "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" \
	-- src/scripts/train/olmoe-1B-7B.py \
    $runname \
		--save-folder="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \








