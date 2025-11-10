# default command explanations:

# the first name is the name appear in beaker
# for more details, do `python -m olmo_core.launch.beaker --help`

# basically it's running `src/examples/llm/train.py`
# the first config is a run name (used for save_folder, wandb name, etc)
# for more details, `python src/examples/llm/train.py olmo1B-pretrain-01 --dry-run`

# -- trainer.load_path if you want to load from another model

# when the config is a class, we could either use a json string or set individual value
# e.g., `--trainer.hard_stop='value: 100, unit: steps'` or 
#       `--trainer.hard_stop.value=100 --trainer.hard_stop.unit=steps`

##############################################################
document_expert_pool=8

runname="twolevel-${document_expert_pool}_1b7b_128experts_olmoe-mix_130B_1110"

torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
    $runname \
		--save-folder="/root/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
		--dataset.mix=arc-easy-train \
		--work-dir="/root/ryanwang/dataset-cache" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
		--model.block.feed_forward_moe.num_experts=16 \
#		--dataset.generate_doc_lengths=true \
#		--model.block.attention.backend=flash_2 \
		--model-type="two-level" \
		--document-expert-pool=${document_expert_pool} \

#python -m olmo_core.launch.beaker \
#  --name $runname \
#	--gpus 8 \
#  --nodes 4 \
#	--weka=oe-training-default \
#  --shared-filesystem \
#	--workspace ai2/flex2 \
#	--cluster ai2/jupiter \
#  --is_private_repo \
#	--preemptible \
#	--allow-dirty \
#	--priority urgent \
#	--env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
#	-- src/scripts/train/olmoe-1B-7B_fsl.py \
#    $runname \
#		--save-folder="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
#		--dataset.mix=OLMoE-mix-0824 \
#		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
#		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
#		--model.block.feed_forward_moe.num_experts=16 \
#		--dataset.generate_doc_lengths=true \
#		--model.block.attention.backend=flash_2 \
#		--model-type="twolevel"
#		--document-expert-pool=${document_expert_pool} \








