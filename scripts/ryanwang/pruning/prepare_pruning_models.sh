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

run_configs=(
  "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115|prune_keep_k=32"
#  "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115|prune_keep_k=8"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103|prune_keep_k=32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110|prune_keep_k=32"
)
#model_name="moe_1b7b_olmoe-mix"
step="step30995"

# these should correspond to activation files
task_names=(
  "arc_easy_rc_validation"
#  "arc_challenge_rc_validation"
#  "boolq_rc_validation"
#  "csqa_rc_validation"
#  "hellaswag_rc_validation"
#  "openbookqa_rc_validation"
#  "piqa_rc_validation"
#  "socialiqa_rc_validation"
#  "winogrande_rc_validation"
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

for run_config in "${run_configs[@]}"; do
    # Split the run_config into model_name and prune_keep_k
    model_name=${run_config%%|*}

    # extract prune_keep_k value
    prune_keep_k=${run_config##*|}
    prune_keep_k="${prune_keep_k#prune_keep_k=}"

    base_model="${BASE_OUTPUT_DIR}/models/${model_name}/${step}"

    echo "Using base model: $base_model"

    for task_name in "${task_names[@]}"; do
        echo "Processing train task: ${task_name}"

        # this is the prefix of the output task name
        task_prefix=$(get_eval_filename "${task_name}")

        activation_file="${BASE_OUTPUT_DIR}/prune/${model_name}_${step}-hf/${task_prefix}-router.jsonl"

        runname="pruneprepmodel-${model_name}_${step}_${task_prefix}_keepk${prune_keep_k}"
        # limit runname to 128 characters, take first 25 and last 75
        runname=$(echo $runname | cut -c1-35)_$(echo $runname | rev | cut -c1-65 | rev)

        # for debugging
        echo "Run name: $runname"
        echo "Base model: $base_model"
        echo "save path" "${base_model}_${task_prefix}_keepk${prune_keep_k}/"
        echo "Prune keep k: $prune_keep_k"
        echo "Activation file: $activation_file"

        gantry run \
            --name $job_name \
            --weka oe-training-default:/weka/oe-training-default \
            --install "pip install -e \".[all]\"" \
            --budget ai2/oceo \
            --workspace ai2/flex2 \
            --cluster $CLUSTER \
            --priority urgent \
            --gpus 1 \
            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
            -- \
            bash -c "python src/scripts/eval/prune_moe_checkpoint.py \
                --checkpoint_path "$base_model" \
                --save_path "${base_model}_${out_dir}/" \
                --prune_keep_k ${prune_keep_k} \
                --activation_file $activation_file \
            "


    done
done






