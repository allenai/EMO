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

model_name="olmoe-pretrain-mose-natural-1022"
step="step30995"
task="arc-easy-mc"
prune_keep_k=32

#base_model="/weka/oe-training-default/ryanwang/phdbrainstorm/models/${model_name}/${step}"
#activation_file="/weka/oe-training-default/ryanwang/phdbrainstorm/evals/weka_oe-training-default_ryanwang_phdbrainstorm_models_${model_name}_${step}-hf/${task}-router.jsonl"

base_model="/root/ryanwang/phdbrainstorm/models/${model_name}/${step}"
activation_file="/root/ryanwang/phdbrainstorm/evals/weka_oe-training-default_ryanwang_phdbrainstorm_models_${model_name}_${step}-hf/${task}-router.jsonl"

runname="olmoe-finetune-${task}"

torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_finetune.py \
    $runname \
		--save-folder="${base_model}/$runname" \
		--dataset.mix=arc-easy-train \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.max_duration='{value: 3, unit: epochs}' \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
		--load_path=$base_model \
		--activation_file=$activation_file \
		--prune_keep_k=$prune_keep_k \

#python -m olmo_core.launch.beaker \
#  --name $runname \
#	--gpus 4 \
#  --nodes 1 \
#  --is_private_repo \
#	--weka=oe-training-default \
#  --shared-filesystem \
#	--workspace ai2/flex2 \
#	--cluster ai2/jupiter \
#	--preemptible \
#	--allow-dirty \
#	--priority urgent \
#	--env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" \
#	-- src/scripts/train/olmoe-1B-7B_finetune.py \
#    $runname \
#		--save-folder="${base_model}/$runname" \
#		--dataset.mix=arc-easy-train \
#		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
#		--trainer.max_duration='{value: 3, unit: epochs}' \
#		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
#		--load_path=$base_model \
#		--activation_file=$activation_file \
#		--prune_keep_k=$prune_keep_k \








