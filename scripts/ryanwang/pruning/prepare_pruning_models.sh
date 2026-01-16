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
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121|prune_keep_k=32"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121|prune_keep_k=8"

#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203|prune_keep_k=32"
#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203|prune_keep_k=8"


#    "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205|prune_keep_k=32"
#    "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205|prune_keep_k=8"

#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123|prune_keep_k=32"

#    "twoleveltoppbatchlb_1b14b_topp-0.35_max-64_min-1_lb-1e-1_1222|prune_keep_k=32"
#    "twoleveltoppbatchlb_1b14b_topp-0.35_max-64_min-1_lb-1e-1_1222|prune_keep_k=16"
#    "twoleveltoppbatchlb_1b14b_topp-0.35_max-64_min-1_lb-1e-1_1222|prune_keep_k=64"

#    "moe_1b35b_320experts_lb-1e-1_1214|prune_keep_k=128"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_poolsched-lineardecay2000_1217|prune_keep_k=128"
#    "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216|prune_keep_k=128"
#    "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216|prune_keep_k=32"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_1219|prune_keep_k=128"

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
#  "task-gsm8k_generation_validation_0shot"
#  "task-coqa_validation_0shot"
#  "task-coqa_full_validation_0shot"
#  "task-squad_validation_0shot"

##   MMLU
  "task-mmlu_biology_rc_validation"
  "task-mmlu_business_rc_validation"
  "task-mmlu_chemistry_rc_validation"
  "task-mmlu_computer_science_rc_validation"
  "task-mmlu_culture_rc_validation"
  "task-mmlu_economics_rc_validation"
  "task-mmlu_engineering_rc_validation"
  "task-mmlu_geography_rc_validation"
  "task-mmlu_health_rc_validation"
  "task-mmlu_history_rc_validation"
  "task-mmlu_law_rc_validation"
  "task-mmlu_math_rc_validation"
  "task-mmlu_other_rc_validation"
  "task-mmlu_philosophy_rc_validation"
  "task-mmlu_physics_rc_validation"
  "task-mmlu_politics_rc_validation"
  "task-mmlu_psychology_rc_validation"

#  "task-mmlu_abstract_algebra_rc_validation"
#  "task-mmlu_anatomy_rc_validation"
#  "task-mmlu_astronomy_rc_validation"
#  "task-mmlu_business_ethics_rc_validation"
#  "task-mmlu_clinical_knowledge_rc_validation"
#  "task-mmlu_college_biology_rc_validation"
#  "task-mmlu_college_chemistry_rc_validation"
#  "task-mmlu_college_computer_science_rc_validation"
#  "task-mmlu_college_mathematics_rc_validation"
#  "task-mmlu_college_medicine_rc_validation"
#  "task-mmlu_college_physics_rc_validation"
#  "task-mmlu_computer_security_rc_validation"
#  "task-mmlu_conceptual_physics_rc_validation"
#  "task-mmlu_econometrics_rc_validation"
#  "task-mmlu_electrical_engineering_rc_validation"
#  "task-mmlu_elementary_mathematics_rc_validation"
#  "task-mmlu_formal_logic_rc_validation"
#  "task-mmlu_global_facts_rc_validation"
#  "task-mmlu_high_school_biology_rc_validation"
#  "task-mmlu_high_school_chemistry_rc_validation"
#  "task-mmlu_high_school_computer_science_rc_validation"
#  "task-mmlu_high_school_european_history_rc_validation"
#  "task-mmlu_high_school_geography_rc_validation"
#  "task-mmlu_high_school_government_and_politics_rc_validation"
#  "task-mmlu_high_school_macroeconomics_rc_validation"
#  "task-mmlu_high_school_mathematics_rc_validation"
#  "task-mmlu_high_school_microeconomics_rc_validation"
#  "task-mmlu_high_school_physics_rc_validation"
#  "task-mmlu_high_school_psychology_rc_validation"
#  "task-mmlu_high_school_statistics_rc_validation"
#  "task-mmlu_high_school_us_history_rc_validation"
#  "task-mmlu_high_school_world_history_rc_validation"
#  "task-mmlu_human_aging_rc_validation"
#  "task-mmlu_human_sexuality_rc_validation"
#  "task-mmlu_international_law_rc_validation"
#  "task-mmlu_jurisprudence_rc_validation"
#  "task-mmlu_logical_fallacies_rc_validation"
#  "task-mmlu_machine_learning_rc_validation"
#  "task-mmlu_management_rc_validation"
#  "task-mmlu_marketing_rc_validation"
#  "task-mmlu_medical_genetics_rc_validation"
#  "task-mmlu_miscellaneous_rc_validation"
#  "task-mmlu_moral_disputes_rc_validation"
#  "task-mmlu_moral_scenarios_rc_validation"
#  "task-mmlu_nutrition_rc_validation"
#  "task-mmlu_philosophy_rc_validation"
#  "task-mmlu_prehistory_rc_validation"
#  "task-mmlu_professional_accounting_rc_validation"
#  "task-mmlu_professional_law_rc_validation"
#  "task-mmlu_professional_medicine_rc_validation"
#  "task-mmlu_professional_psychology_rc_validation"
#  "task-mmlu_public_relations_rc_validation"
#  "task-mmlu_security_studies_rc_validation"
#  "task-mmlu_sociology_rc_validation"
#  "task-mmlu_us_foreign_policy_rc_validation"
#  "task-mmlu_virology_rc_validation"
#  "task-mmlu_world_religions_rc_validation"

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






