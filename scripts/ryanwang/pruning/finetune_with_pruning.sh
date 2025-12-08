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
BASE_OUTPUT_DIR="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE"
#BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE"

model_names=(
#  "moe_1b14b_128experts_olmoe-mix_130B_1117"
#  "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115"

   "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121"
#   "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123"
#   "twolevelsamplingnolb-32_1b14b_stability_1127"
)
#model_name="moe_1b7b_olmoe-mix"
step="step30995"
num_checkpoints=5

# this is used for ablations
variation="newdefault_lr-4e-4"

#experiment_tag="pruned_finetuning"
experiment_tag="pruned_finetuning_ablate"

variation_flags=""
# Define variation-specific settings
if [ "$variation" == "noloadoptim" ]; then
    variation_flags="--trainer.load_optim_state=false --trainer.load_trainer_state=false"
elif [ "$variation" == "newdefault_lr-4e-5" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-5"
elif [ "$variation" == "newdefault_lr-4e-4" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-4"
else
    echo "Warning: Unknown variation '$variation'. Using default settings."
    variation_flags=""
fi

# first argument is which validation used for pruning, second is training dataset
task_configs=(
#  "task-arc_easy_rc_validation_keepk32|arc_easy:rc_train::olmes"
  "task-arc_challenge_rc_validation_keepk32|arc_challenge:rc_train::olmes"
  "task-boolq_rc_validation_keepk32|boolq:rc_train::olmes"
  "task-csqa_rc_validation_keepk32|csqa:rc_train::olmes"
  "task-hellaswag_rc_validation_keepk32|hellaswag:rc_train::olmes"
  "task-openbookqa_rc_validation_keepk32|openbookqa:rc_train::olmes"
  "task-piqa_rc_validation_keepk32|piqa:rc_train::olmes"
  "task-socialiqa_rc_validation_keepk32|socialiqa:rc_train::olmes"
  "task-winogrande_rc_validation_keepk32|winogrande:rc_train::olmes"

  # following is depricated for now
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

for model_name in "${model_names[@]}"; do
    for task_config in "${task_configs[@]}"; do
        # Split the task_config into pruned_model_name and train_task_name
        pruned_model_name=${task_config%%|*}
        train_task_name=${task_config##*|}

        echo "Processing train task: $train_task_name"

        # this is the prefix of the output task name
        task_prefix=$(get_eval_filename "$train_task_name")

        # we now tokenize the file
        tokenizer_name="allenai/OLMo-2-1124-7B"
        data_folder="${BASE_OUTPUT_DIR}/prune/${task_prefix}-tokenized"
        echo "data_folder : $data_folder"

        # Collect the corresponding dataset paths. Can make this assumption given we have a warning in tokenize script
        dataset_paths="${data_folder}/part-0-00000.npy"
        label_mask_paths="${data_folder}/part-0-00000_mask.npy"

        runname="${model_name}/${step}_${pruned_model_name}_finetune-${task_prefix}"
        wandb_name=${runname}
        # limit runname to 128 characters, take first 25 and last 75
        runname=$(echo $runname | cut -c1-35)_$(echo $runname | rev | cut -c1-65 | rev)
#        runname=$(echo $runname | rev | cut -c1-100 | rev)

        out_dir="finetune-${task_prefix}"

        base_model="${BASE_OUTPUT_DIR}/models/${model_name}/${step}_${pruned_model_name}"

        echo "Using base model: $base_model"

        # get the prune_keep_k value from pruned_model_name
        prune_keep_k=${pruned_model_name##*_keepk}

        # for debugging
        echo "Run name: $runname"
        echo "Dataset paths: ${dataset_paths}"
        echo "Label mask paths: ${label_mask_paths}"
        echo "Base model: $base_model"
        echo "Save folder: ${base_model}/${out_dir}"
        echo "Prune keep k: $prune_keep_k"

        # define

#        torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_finetune.py \
#            $runname \
#        		--save-folder="${base_model}/${variation}/${out_dir}" \
#            --dataset.paths="[${dataset_paths}]" \
#            --dataset.label_mask_paths="[${label_mask_paths}]" \
#            --work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
#        		--trainer.max_duration='{value: 3, unit: epochs}' \
#        		--trainer.callbacks.wandb="{enabled: false, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
#        		--train_module.compile_model=false \
#        		--load_path=$base_model \
#        		--num_checkpoints=$num_checkpoints \
#        		--model.block.feed_forward_moe.num_experts=${prune_keep_k} \
#            --model.block.name="moe" \
#            --model.block.attention.qk_norm=null \
#            --trainer.load_optim_state=false \
#            --trainer.load_trainer_state=false \
#            --global_batch_size=32 \
#            $variation_flags

        # throw error if not load_optim_state and load_trainer_state are false in variation_flags
        if [[ $variation != *"newdefault"* ]]; then
            echo "Error: must be of newdefault type (i.e reinitialize optim, masked finetuning)"
            exit 1
        fi

        python -m olmo_core.launch.beaker \
          --name $runname \
          --gpus 8 \
          --nodes 1 \
          --is_private_repo \
          --weka=oe-training-default \
          --shared-filesystem \
          --workspace ai2/flex2 \
          --cluster ai2/jupiter \
          --preemptible \
          --allow-dirty \
          --priority urgent \
          --no-follow \
          --env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" \
          -- src/scripts/train/olmoe-1B-7B_finetune.py \
            $runname \
            --save-folder="${base_model}/${variation}/${out_dir}" \
            --dataset.paths="[${dataset_paths}]" \
            --dataset.label_mask_paths="[${label_mask_paths}]" \
            --work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
            --trainer.max_duration='{value: 3, unit: epochs}' \
            --trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${wandb_name}, tags: [${task_prefix:0:64}, ${model_name:0:64}, ${pruned_model_name}, ${experiment_tag}]}" \
            --load_path=$base_model \
            --num_checkpoints=$num_checkpoints \
            --model.block.feed_forward_moe.num_experts=${prune_keep_k} \
            --model.block.name="moe" \
		        --model.block.attention.qk_norm=null \
		        --trainer.load_optim_state=false \
		        --trainer.load_trainer_state=false \
		        --global_batch_size=32 \
            $variation_flags

    #        --dataset.label_mask_paths="[${label_mask_paths}]" \

    done
done






