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

# more details: https://olmo-core.readthedocs.io/en/researcher-quick-start/guides/all_in_one_for_researchers.html

# this is training a llama2_271M (adjustable through `--model-factory`)

runname="olmo2_7B_pubmed_continual-anneal_vanilla"
python -m olmo_core.launch.beaker \
	--name $runname \
	--gpus 8 \
	--nodes 2 \
	--budget ai2/oe-base \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--priority urgent \
	--preemptible \
	--torchrun \
	--weka=oe-training-default \
	--shared-filesystem \
	--env-secret HF_TOKEN=RYAN_HF_TOKEN \
  --env-secret WANDB_API_KEY=RYAN_WANDB_API_KEY \
	-- src/scripts/train/olmo2-7b_continual-anneal.py \
		$runname \
		--model-factory=olmo2_7B \
		--sequence-length=4096 \
		--trainer.save_folder=/weka/oe-training-default/ryanwang/phdbrainstorm/models/$runname \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmo2_7B, name: ${runname}}" \



#	--allow-dirty \
#		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#		--trainer.hard_stop='{value: 100, unit: steps}' \


#CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=1 src/scripts/train/olmo2-7b_continual.py \
#		$runname \
#		--model-factory=olmo2_190M \
#		--sequence-length=1024 \
#		--trainer.save_folder=/root/phdbrainstorm/models/$runname \
#		--work-dir="/root/dataset-cache" \
#		--trainer.callbacks.wandb='{enabled: true, entity: ryanyxw, project: olmo2_7B, name: $runname}' \
#		--trainer.hard_stop='{value: 100, unit: steps}' \