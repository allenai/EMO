# PARENT: "dense_1b_olmoe-mix_1028.sh"
# DESCRIPTION:
#     - Different from parent by adding dataset filtering
#     - Re-run to also collect in-loop evals (1028 version did not have in-loop evals setup)
# STATUS: DEPRICATED
#     - Decuded to use prenorm + noqknorm instead (dense_1b_olmoe-mix_prenorm_noqknorm_1123.sh)
##############################################################

runname="dense_1b_olmoe-mix_1119"
python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus 8 \
  --nodes 8 \
  --is_private_repo \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--preemptible \
	--allow-dirty \
	--priority urgent \
	--env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" \
	-- src/scripts/train/olmo2-1B.py \
    $runname \
		--save-folder="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \









