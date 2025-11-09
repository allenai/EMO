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
BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE"

model_name="moe_1b7b_128experts_olmoe-mix_130B_1103"
step="step30995"
prune_keep_k=32

base_model="${BASE_OUTPUT_DIR}/models/${model_name}/${step}"

train_task_names=(
#  "arc_easy:rc_train_0shot::olmes"
  "arc_challenge:rc_train_0shot::olmes"
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
    destination="${BASE_OUTPUT_DIR}/prune/${task_prefix}-tokenized"
    echo "destination folder: $destination"

    # Collect the corresponding dataset paths
    dataset_paths=($(ls ${destination}/*.npy | grep -v mask.npy))
    label_mask_paths=($(ls ${destination}/*_mask.npy))

    # Convert label_mask_paths array to a comma-separated string
    IFS=','
    dataset_paths_str="${dataset_paths[*]}"
    label_mask_paths_str="${label_mask_paths[*]}"
    unset IFS

    # swap out all occurences of "train" with "validation" to get validation set
    validation_task_prefix="${task_prefix/train/validation}"
    activation_file="${BASE_OUTPUT_DIR}/prune/${model_name}_${step}-hf/${validation_task_prefix}-router.jsonl"

    runname="${model_name}_${step}_finetune_${task_prefix}_keepk${prune_keep_k}"

    # for debugging
    echo "Run name: $runname"
    echo "Dataset paths: ${dataset_paths[@]}"
    echo "Label mask paths: ${label_mask_paths[@]}"
    echo "Base model: $base_model"
    echo "Prune keep k: $prune_keep_k"
    echo "Activation file: $activation_file"

    torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_finetune.py \
        $runname \
    		--save-folder="${base_model}/$runname" \
        --dataset.paths="[${dataset_paths_str}]" \
        --dataset.label_mask_paths="[${label_mask_paths_str}]" \
        --work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
    		--trainer.max_duration='{value: 3, unit: epochs}' \
    		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
    		--load_path=$base_model \
    		--activation_file=$activation_file \
    		--prune_keep_k=$prune_keep_k \

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
#      --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" \
#      -- src/scripts/train/olmoe-1B-7B_finetune.py \
#        $runname \
#        --save-folder="${base_model}/$runname" \
#        --dataset.paths="[${dataset_paths_str}]" \
#        --dataset.label_mask_paths="[${label_mask_paths_str}]" \
#        --work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
#        --trainer.max_duration='{value: 3, unit: epochs}' \
#        --trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
#        --load_path=$base_model \
#        --activation_file=$activation_file \
#        --prune_keep_k=$prune_keep_k \


done






