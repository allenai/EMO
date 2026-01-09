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
   "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121"
#   "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203"

#   "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205"

#   "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123"
#   "twolevelsamplingnolb-32_1b14b_stability_1127"

#    "twoleveltoppbatchlb_1b14b_topp-0.35_max-64_min-1_lb-1e-1_1222"


#    "moe_1b35b_320experts_lb-1e-1_1214"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_poolsched-lineardecay2000_1217"
#    "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_1219"
)
#model_name="moe_1b7b_olmoe-mix"
step="step30995"
num_checkpoints=5

# this is used for ablations
variation="newdefault_lr-4e-5_bs-8"

experiment_tag="pruned_finetuning"
#experiment_tag="pruned_finetuning_ablate"

variation_flags=""
# Define variation-specific settings
if [ "$variation" == "noloadoptim" ]; then
    variation_flags="--trainer.load_optim_state=false --trainer.load_trainer_state=false"
elif [ "$variation" == "newdefault_lr-4e-5" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-5 --global_batch_size=32"
elif [ "$variation" == "newdefault_lr-4e-5_bs-128" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-5 --global_batch_size=128"
elif [ "$variation" == "newdefault_lr-4e-5_bs-8" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-5 --global_batch_size=8"
elif [ "$variation" == "newdefault_lr-4e-4" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-4 --global_batch_size=32"
elif [ "$variation" == "newdefault_lr-4e-6" ]; then
    # reinitialize optim and use masked finetuning (should be checked)
    variation_flags="--train_module.optim.lr=4e-6 --global_batch_size=32"
elif [ "$variation" == "newdefault_lr-4e-6_bs-128" ]; then
    # reinitialize optim and use masked finetuning (should be checked) and batch size of 128 (should be checked)
    variation_flags="--train_module.optim.lr=4e-6 --global_batch_size=128"
else
    echo "error: Unknown variation '$variation'. "
    exit 1
fi

# first argument is which validation used for pruning, second is training dataset
task_configs=(
#  "task-arc_easy_rc_validation_keepk32|arc_easy:rc_train::olmes"
#  "task-arc_challenge_rc_validation_keepk32|arc_challenge:rc_train::olmes"
#  "task-boolq_rc_validation_keepk32|boolq:rc_train::olmes"
#  "task-csqa_rc_validation_keepk32|csqa:rc_train::olmes"
#  "task-hellaswag_rc_validation_keepk32|hellaswag:rc_train::olmes"
#  "task-openbookqa_rc_validation_keepk32|openbookqa:rc_train::olmes"
#  "task-piqa_rc_validation_keepk32|piqa:rc_train::olmes"
#  "task-socialiqa_rc_validation_keepk32|socialiqa:rc_train::olmes"
#  "task-winogrande_rc_validation_keepk32|winogrande:rc_train::olmes"
#  "task-synthea_rc_validation_0shot_keepk32|synthea:rc_train_0shot::olmes"
#  "task-gsm8k_generation_validation_0shot_keepk32|gsm8k_generation:train_0shot::olmes"

#  "task-gsm8k_generation_validation_0shot_keepk128|gsm8k_generation:train_0shot::olmes"
#  "task-coqa_validation_0shot_keepk32|coqa:train_0shot::olmes"
#  "task-squad_validation_0shot_keepk32|squad:train_0shot::olmes"

#  "task-mmlu_abstract_algebra_rc_validation_keepk32|mmlu_abstract_algebra:rc_train::olmes"
#  "task-mmlu_anatomy_rc_validation_keepk32|mmlu_anatomy:rc_train::olmes"
#  "task-mmlu_astronomy_rc_validation_keepk32|mmlu_astronomy:rc_train::olmes"
#  "task-mmlu_business_ethics_rc_validation_keepk32|mmlu_business_ethics:rc_train::olmes"
  "task-mmlu_clinical_knowledge_rc_validation_keepk32|mmlu_clinical_knowledge:rc_train::olmes"
  "task-mmlu_college_biology_rc_validation_keepk32|mmlu_college_biology:rc_train::olmes"
  "task-mmlu_college_chemistry_rc_validation_keepk32|mmlu_college_chemistry:rc_train::olmes"
  "task-mmlu_college_computer_science_rc_validation_keepk32|mmlu_college_computer_science:rc_train::olmes"
  "task-mmlu_college_mathematics_rc_validation_keepk32|mmlu_college_mathematics:rc_train::olmes"
  "task-mmlu_college_medicine_rc_validation_keepk32|mmlu_college_medicine:rc_train::olmes"
  "task-mmlu_college_physics_rc_validation_keepk32|mmlu_college_physics:rc_train::olmes"
#  "task-mmlu_computer_security_rc_validation_keepk32|mmlu_computer_security:rc_train::olmes"
#  "task-mmlu_conceptual_physics_rc_validation_keepk32|mmlu_conceptual_physics:rc_train::olmes"
#  "task-mmlu_econometrics_rc_validation_keepk32|mmlu_econometrics:rc_train::olmes"
#  "task-mmlu_electrical_engineering_rc_validation_keepk32|mmlu_electrical_engineering:rc_train::olmes"
#  "task-mmlu_elementary_mathematics_rc_validation_keepk32|mmlu_elementary_mathematics:rc_train::olmes"
#  "task-mmlu_formal_logic_rc_validation_keepk32|mmlu_formal_logic:rc_train::olmes"
#  "task-mmlu_global_facts_rc_validation_keepk32|mmlu_global_facts:rc_train::olmes"
#  "task-mmlu_high_school_biology_rc_validation_keepk32|mmlu_high_school_biology:rc_train::olmes"
#  "task-mmlu_high_school_chemistry_rc_validation_keepk32|mmlu_high_school_chemistry:rc_train::olmes"
#  "task-mmlu_high_school_computer_science_rc_validation_keepk32|mmlu_high_school_computer_science:rc_train::olmes"
#  "task-mmlu_high_school_european_history_rc_validation_keepk32|mmlu_high_school_european_history:rc_train::olmes"
#  "task-mmlu_high_school_geography_rc_validation_keepk32|mmlu_high_school_geography:rc_train::olmes"
#  "task-mmlu_high_school_government_and_politics_rc_validation_keepk32|mmlu_high_school_government_and_politics:rc_train::olmes"
#  "task-mmlu_high_school_macroeconomics_rc_validation_keepk32|mmlu_high_school_macroeconomics:rc_train::olmes"
#  "task-mmlu_high_school_mathematics_rc_validation_keepk32|mmlu_high_school_mathematics:rc_train::olmes"
#  "task-mmlu_high_school_microeconomics_rc_validation_keepk32|mmlu_high_school_microeconomics:rc_train::olmes"
#  "task-mmlu_high_school_physics_rc_validation_keepk32|mmlu_high_school_physics:rc_train::olmes"
#  "task-mmlu_high_school_psychology_rc_validation_keepk32|mmlu_high_school_psychology:rc_train::olmes"
#  "task-mmlu_high_school_statistics_rc_validation_keepk32|mmlu_high_school_statistics:rc_train::olmes"
#  "task-mmlu_high_school_us_history_rc_validation_keepk32|mmlu_high_school_us_history:rc_train::olmes"
#  "task-mmlu_high_school_world_history_rc_validation_keepk32|mmlu_high_school_world_history:rc_train::olmes"
#  "task-mmlu_human_aging_rc_validation_keepk32|mmlu_human_aging:rc_train::olmes"
#  "task-mmlu_human_sexuality_rc_validation_keepk32|mmlu_human_sexuality:rc_train::olmes"
#  "task-mmlu_international_law_rc_validation_keepk32|mmlu_international_law:rc_train::olmes"
#  "task-mmlu_jurisprudence_rc_validation_keepk32|mmlu_jurisprudence:rc_train::olmes"
#  "task-mmlu_logical_fallacies_rc_validation_keepk32|mmlu_logical_fallacies:rc_train::olmes"
#  "task-mmlu_machine_learning_rc_validation_keepk32|mmlu_machine_learning:rc_train::olmes"
#  "task-mmlu_management_rc_validation_keepk32|mmlu_management:rc_train::olmes"
#  "task-mmlu_marketing_rc_validation_keepk32|mmlu_marketing:rc_train::olmes"
#  "task-mmlu_medical_genetics_rc_validation_keepk32|mmlu_medical_genetics:rc_train::olmes"
#  "task-mmlu_miscellaneous_rc_validation_keepk32|mmlu_miscellaneous:rc_train::olmes"
#  "task-mmlu_moral_disputes_rc_validation_keepk32|mmlu_moral_disputes:rc_train::olmes"
#  "task-mmlu_moral_scenarios_rc_validation_keepk32|mmlu_moral_scenarios:rc_train::olmes"
#  "task-mmlu_nutrition_rc_validation_keepk32|mmlu_nutrition:rc_train::olmes"
#  "task-mmlu_philosophy_rc_validation_keepk32|mmlu_philosophy:rc_train::olmes"
#  "task-mmlu_prehistory_rc_validation_keepk32|mmlu_prehistory:rc_train::olmes"
#  "task-mmlu_professional_accounting_rc_validation_keepk32|mmlu_professional_accounting:rc_train::olmes"
#  "task-mmlu_professional_law_rc_validation_keepk32|mmlu_professional_law:rc_train::olmes"
#  "task-mmlu_professional_medicine_rc_validation_keepk32|mmlu_professional_medicine:rc_train::olmes"
#  "task-mmlu_professional_psychology_rc_validation_keepk32|mmlu_professional_psychology:rc_train::olmes"
#  "task-mmlu_public_relations_rc_validation_keepk32|mmlu_public_relations:rc_train::olmes"
#  "task-mmlu_security_studies_rc_validation_keepk32|mmlu_security_studies:rc_train::olmes"
#  "task-mmlu_sociology_rc_validation_keepk32|mmlu_sociology:rc_train::olmes"
#  "task-mmlu_us_foreign_policy_rc_validation_keepk32|mmlu_us_foreign_policy:rc_train::olmes"
#  "task-mmlu_virology_rc_validation_keepk32|mmlu_virology:rc_train::olmes"
#  "task-mmlu_world_religions_rc_validation_keepk32|mmlu_world_religions:rc_train::olmes"


#
#  "task-arc_easy_rc_validation_keepk8|arc_easy:rc_train::olmes"
#  "task-arc_challenge_rc_validation_keepk8|arc_challenge:rc_train::olmes"
#  "task-boolq_rc_validation_keepk8|boolq:rc_train::olmes"
#  "task-csqa_rc_validation_keepk8|csqa:rc_train::olmes"
#  "task-hellaswag_rc_validation_keepk8|hellaswag:rc_train::olmes"
#  "task-openbookqa_rc_validation_keepk8|openbookqa:rc_train::olmes"
#  "task-piqa_rc_validation_keepk8|piqa:rc_train::olmes"
#  "task-socialiqa_rc_validation_keepk8|socialiqa:rc_train::olmes"
#  "task-winogrande_rc_validation_keepk8|winogrande:rc_train::olmes"

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

        runname="${model_name}_${step}_${pruned_model_name}_finetune-${task_prefix}_${variation}"
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
#		        --model-type="masked-finetune" \
#            --model.block.name="moe" \
#            --model.block.attention.qk_norm=null \
#        		--model.block.feed_forward_moe.num_experts=${prune_keep_k} \
#            --trainer.load_optim_state=false \
#            --trainer.load_trainer_state=false \
#            --global_batch_size=32 \
#            $variation_flags

        # throw error if not load_optim_state and load_trainer_state are false in variation_flags
        if [[ $variation != *"newdefault"* ]]; then
            echo "Error: must be of newdefault type (i.e reinitialize optim, masked finetuning)"
            exit 1
        fi

        # for mmlu, we use less gpus since we have a pretty small batch size
        if [[ $task_config == *"mmlu"* ]]; then
            num_gpus=4
        else
            num_gpus=8
        fi

        python -m olmo_core.launch.beaker \
          --name $runname \
          --gpus $num_gpus \
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
		        --model-type="masked-finetune" \
            --model.block.name="moe" \
		        --model.block.attention.qk_norm=null \
            --model.block.feed_forward_moe.num_experts=${prune_keep_k} \
		        --trainer.load_optim_state=false \
		        --trainer.load_trainer_state=false \
            $variation_flags

        sleep 30

    #        --dataset.label_mask_paths="[${label_mask_paths}]" \

    done
done






