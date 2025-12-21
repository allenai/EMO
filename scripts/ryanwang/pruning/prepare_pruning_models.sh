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
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121|prune_keep_k=32"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121|prune_keep_k=8"

#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203|prune_keep_k=32"
#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203|prune_keep_k=8"


#    "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205|prune_keep_k=32"
#    "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205|prune_keep_k=8"

#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123|prune_keep_k=32"

#    "moe_1b35b_320experts_lb-1e-1_1214|prune_keep_k=128"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_poolsched-lineardecay2000_1217|prune_keep_k=128"
    "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216/step30995-hf"
    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_1219/step30995-hf"

#    "twolevelsamplingnolb-32_1b14b_stability_1127|prune_keep_k=32"
)

#model_name="moe_1b7b_olmoe-mix"
step="step30995"

# these should correspond to activation files
validation_task_names=(
#  "task-arc_easy_rc_validation"
#  "task-arc_challenge_rc_validation"
#  "task-boolq_rc_validation"
#  "task-csqa_rc_validation"
#  "task-hellaswag_rc_validation"
#  "task-openbookqa_rc_validation"
#  "task-piqa_rc_validation"
#  "task-socialiqa_rc_validation"
#  "task-winogrande_rc_validation"
#  "task-synthea_rc_validation_0shot"
  "task-gsm8k_generation_validation_0shot"

##   MMLU
#  "mmlu_rc:rc_train_0shot::olmes"
#
##   GSM8K
#  "gsm8k:perplexity_train_0shot::olmes"
)

for run_config in "${run_configs[@]}"; do
    # Split the run_config into model_name and prune_keep_k
    model_name=${run_config%%|*}

    # extract prune_keep_k value
    prune_keep_k=${run_config##*|}
    prune_keep_k="${prune_keep_k#prune_keep_k=}"

    base_model="${BASE_OUTPUT_DIR}/models/${model_name}/${step}"

    echo "Using base model: $base_model"

    for validation_task_name in "${validation_task_names[@]}"; do
        echo "Processing train task: ${validation_task_name}"

        activation_file="${BASE_OUTPUT_DIR}/prune/${model_name}_${step}-hf/${validation_task_name}-router.jsonl"

        runname="pruneprepmodel-${model_name}_${step}_${validation_task_name}_keepk${prune_keep_k}"
        # limit runname to 128 characters, take first 25 and last 75
        runname=$(echo $runname | cut -c1-35)_$(echo $runname | rev | cut -c1-65 | rev)

        # for debugging
        echo "Run name: $runname"
        echo "Base model: $base_model"
        echo "save path" "${base_model}_${validation_task_name}_keepk${prune_keep_k}"
        echo "Prune keep k: $prune_keep_k"
        echo "Activation file: $activation_file"

        python -m olmo_core.launch.beaker \
          --name $runname \
          --gpus 1 \
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
          -- src/scripts/eval/prune_moe_checkpoint.py \
            --checkpoint_path "$base_model" \
            --save_path "${base_model}_${validation_task_name}_keepk${prune_keep_k}" \
            --prune_keep_k ${prune_keep_k} \
            --activation_file $activation_file \

    done
done






