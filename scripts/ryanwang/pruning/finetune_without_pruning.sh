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
#BASE_OUTPUT_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE"
BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE"

model_name="dense_1b_olmoe-mix_prenorm_noqknorm_1123"
#model_name="moe_1b7b_olmoe-mix"
step="step30995"
num_checkpoints=5

# this is used for ablations
variation="masked"

variation_flags=""
# Define variation-specific settings
if [ "$variation" == "noloadoptim" ]; then
    # Variation: No optimizer state loading
    # Changes: Initialize optimizer from scratch instead of loading from checkpoint.
    variation_flags="--trainer.load_optim_state=false --trainer.load_trainer_state=false"
elif [ "$variation" == "masked" ]; then
    # Variation: Masked Fine-tuning
    # Changes: Apply a mask to certain model parameters during fine-tuning. TODO change this to be permanent
    variation_flags=""
else
    echo "Warning: Unknown variation '$variation'. Using default settings."
    variation_flags=""
fi

base_model="${BASE_OUTPUT_DIR}/models/${model_name}/${step}"

train_task_names=(
  "arc_easy:rc_train::olmes"
#  "arc_challenge:rc_train::olmes"
#  "boolq:rc_train::olmes"
#  "csqa:rc_train::olmes"
#  "hellaswag:rc_train::olmes"
#  "openbookqa:rc_train::olmes"
#  "piqa:rc_train::olmes"
#  "socialiqa:rc_train::olmes"
#  "winogrande:rc_train::olmes"

#  "arc_easy:rc_train_0shot::olmes"
#  "arc_challenge:rc_train_0shot::olmes"
#  "boolq:rc_train_0shot::olmes"
#  "csqa:rc_train_0shot::olmes"
#  "hellaswag:rc_train_0shot::olmes"
#  "openbookqa:rc_train_0shot::olmes"
#  "piqa:rc_train_0shot::olmes"
#  "socialiqa:rc_train_0shot::olmes"
#  "winogrande:rc_train_0shot::olmes"
#
##   MMLU
#  "mmlu_rc:rc_train_0shot::olmes"
#
##   GSM8K
#  "gsm8k:perplexity_train_0shot::olmes"
)

get_eval_filename() {
    local task_name="$1"

    # Remove everything after and including '::' (if present)
    task_name="${task_name%%::*}"

    # Replace all ':' with '_'
    task_name="${task_name//:/_}"

    # Return the formatted string
    echo "task-${task_name}"
}

for train_task_name in "${train_task_names[@]}"; do
    echo "Processing train task: $train_task_name"

    # this is the prefix of the output task name
    task_prefix=$(get_eval_filename "$train_task_name")

    # we now tokenize the file
    tokenizer_name="allenai/OLMo-2-1124-7B"
    data_folder="${BASE_OUTPUT_DIR}/prune/${task_prefix}-tokenized"
    echo "data_folder folder: $data_folder"

    # Collect the corresponding dataset paths. Can make this assumption given we have a warning in tokenize script
    dataset_paths="${data_folder}/part-0-00000.npy"
    label_mask_paths="${data_folder}/part-0-00000_mask.npy"

    runname="${model_name}_${step}_finetune_${task_prefix}"
    # limit runname to 128 characters, take first 25 and last 75
    runname=$(echo $runname | cut -c1-35)_$(echo $runname | rev | cut -c1-65 | rev)

    out_dir="finetune-${task_prefix}"

    # for debugging
    echo "Run name: $runname"
    echo "Dataset paths: ${dataset_paths[@]}"
    echo "Label mask paths: ${label_mask_paths[@]}"
    echo "Base model: $base_model"
    echo "Output dir: $out_dir"

    # define

    torchrun --nproc-per-node=1 src/scripts/train/olmo2-1B_finetune.py \
        $runname \
    		--save-folder="${base_model}/${variation}/${out_dir}" \
        --dataset.paths="[${dataset_paths}]" \
        --dataset.label_mask_paths="[${label_mask_paths}]" \
        --work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
    		--trainer.max_duration='{value: 3, unit: epochs}' \
    		--trainer.callbacks.wandb="{enabled: false, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
        --train_module.compile_model=false \
    		--model.block.name="default" \
		    --model.block.attention.qk_norm=null \
    		--load_path=$base_model \
    		--num_checkpoints=$num_checkpoints \
        --train_module.optim.lr=2e-5 \
        --trainer.load_optim_state=false \
        --trainer.load_trainer_state=false \
        --data_loader.seed=1 \
    		$variation_flags

    # throw error if not load_optim_state and load_trainer_state are false in variation_flags
    if [[ $variation_flags != *"--trainer.load_optim_state=false"* ]]; then
        echo "Error: --trainer.load_optim_state must be false for finetuning after pruning."
        exit 1
    fi

#    python -m olmo_core.launch.beaker \
#      --name $runname \
#      --gpus 8 \
#      --nodes 1 \
#      --is_private_repo \
#      --weka=oe-training-default \
#      --shared-filesystem \
#      --workspace ai2/flex2 \
#      --cluster ai2/jupiter \
#      --preemptible \
#      --allow-dirty \
#      --priority urgent \
#      --no-follow \
#      --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" \
#      -- src/scripts/train/olmo2-1B_finetune.py \
#        $runname \
#        --save-folder="${base_model}/${variation}/${out_dir}" \
#        --dataset.paths="[${dataset_paths}]" \
#        --work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
#        --trainer.max_duration='{value: 3, unit: epochs}' \
#        --trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}, tags: [finetuning, ${task_prefix:0:64}, ${model_name:0:64}]}" \
#        --model.block.name="default" \
#		    --model.block.attention.qk_norm=null \
#        --load_path=$base_model \
#        --num_checkpoints=$num_checkpoints \
#        $variation_flags

#        --dataset.label_mask_paths="[${label_mask_paths}]" \


done






