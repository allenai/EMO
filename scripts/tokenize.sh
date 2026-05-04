#!/bin/bash

set -e

BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/Emo/prune"

get_eval_filename() {
    local task_name="$1"

    # Remove everything after and including '::' (if present)
    task_name="${task_name%%::*}"

    # Replace all ':' with '_'
    task_name="${task_name//:/_}"

    # Return the formatted string
    echo "task-${task_name}"
}
train_task_names=(
#  "arc_easy:rc_train::olmes"
#  "arc_challenge:rc_train::olmes"
#  "boolq:rc_train::olmes"
#  "csqa:rc_train::olmes"
#  "hellaswag:rc_train::olmes"
#  "openbookqa:rc_train::olmes"
#  "piqa:rc_train::olmes"
#  "socialiqa:rc_train::olmes"
#  "winogrande:rc_train::olmes"
#  "synthea:rc_train_0shot::olmes"
#  "gsm8k_generation:train_0shot::olmes"
#  "coqa:train_0shot::olmes"
#  "coqa_full:train_0shot::olmes"
#  "squad:train_0shot::olmes"


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
#  "mmlu_biology:rc_train::olmes"
#  "mmlu_business:rc_train::olmes"
#  "mmlu_chemistry:rc_train::olmes"
#  "mmlu_computer_science:rc_train::olmes"
#  "mmlu_culture:rc_train::olmes"
#  "mmlu_economics:rc_train::olmes"
#  "mmlu_engineering:rc_train::olmes"
#  "mmlu_geography:rc_train::olmes"
#  "mmlu_health:rc_train::olmes"
#  "mmlu_history:rc_train::olmes"
#  "mmlu_law:rc_train::olmes"
#  "mmlu_math:rc_train::olmes"
#  "mmlu_other:rc_train::olmes"
  "mmlu_philosophy_cat:rc_train::olmes"
#  "mmlu_physics:rc_train::olmes"
#  "mmlu_politics:rc_train::olmes"
#  "mmlu_psychology:rc_train::olmes"

#  "mmlu_abstract_algebra:rc_train::olmes"
#  "mmlu_anatomy:rc_train::olmes"
#  "mmlu_astronomy:rc_train::olmes"
#  "mmlu_business_ethics:rc_train::olmes"
#  "mmlu_clinical_knowledge:rc_train::olmes"
#  "mmlu_college_biology:rc_train::olmes"
#  "mmlu_college_chemistry:rc_train::olmes"
#  "mmlu_college_computer_science:rc_train::olmes"
#  "mmlu_college_mathematics:rc_train::olmes"
#  "mmlu_college_medicine:rc_train::olmes"
#  "mmlu_college_physics:rc_train::olmes"
#  "mmlu_computer_security:rc_train::olmes"
#  "mmlu_conceptual_physics:rc_train::olmes"
#  "mmlu_econometrics:rc_train::olmes"
#  "mmlu_electrical_engineering:rc_train::olmes"
#  "mmlu_elementary_mathematics:rc_train::olmes"
#  "mmlu_formal_logic:rc_train::olmes"
#  "mmlu_global_facts:rc_train::olmes"
#  "mmlu_high_school_biology:rc_train::olmes"
#  "mmlu_high_school_chemistry:rc_train::olmes"
#  "mmlu_high_school_computer_science:rc_train::olmes"
#  "mmlu_high_school_european_history:rc_train::olmes"
#  "mmlu_high_school_geography:rc_train::olmes"
#  "mmlu_high_school_government_and_politics:rc_train::olmes"
#  "mmlu_high_school_macroeconomics:rc_train::olmes"
#  "mmlu_high_school_mathematics:rc_train::olmes"
#  "mmlu_high_school_microeconomics:rc_train::olmes"
#  "mmlu_high_school_physics:rc_train::olmes"
#  "mmlu_high_school_psychology:rc_train::olmes"
#  "mmlu_high_school_statistics:rc_train::olmes"
#  "mmlu_high_school_us_history:rc_train::olmes"
#  "mmlu_high_school_world_history:rc_train::olmes"
#  "mmlu_human_aging:rc_train::olmes"
#  "mmlu_human_sexuality:rc_train::olmes"
#  "mmlu_international_law:rc_train::olmes"
#  "mmlu_jurisprudence:rc_train::olmes"
#  "mmlu_logical_fallacies:rc_train::olmes"
#  "mmlu_machine_learning:rc_train::olmes"
#  "mmlu_management:rc_train::olmes"
#  "mmlu_marketing:rc_train::olmes"
#  "mmlu_medical_genetics:rc_train::olmes"
#  "mmlu_miscellaneous:rc_train::olmes"
#  "mmlu_moral_disputes:rc_train::olmes"
#  "mmlu_moral_scenarios:rc_train::olmes"
#  "mmlu_nutrition:rc_train::olmes"
#  "mmlu_philosophy:rc_train::olmes"
#  "mmlu_prehistory:rc_train::olmes"
#  "mmlu_professional_accounting:rc_train::olmes"
#  "mmlu_professional_law:rc_train::olmes"
#  "mmlu_professional_medicine:rc_train::olmes"
#  "mmlu_professional_psychology:rc_train::olmes"
#  "mmlu_public_relations:rc_train::olmes"
#  "mmlu_security_studies:rc_train::olmes"
#  "mmlu_sociology:rc_train::olmes"
#  "mmlu_us_foreign_policy:rc_train::olmes"
#  "mmlu_virology:rc_train::olmes"
#  "mmlu_world_religions:rc_train::olmes"
#
##   GSM8K
#  "gsm8k:perplexity_train_0shot::olmes"
)

for train_task_name in "${train_task_names[@]}"; do
    echo "Processing train task: $train_task_name"

    # this is the prefix of the output task name
    task_prefix=$(get_eval_filename "$train_task_name")
    processed_train_file="${task_prefix}-processed.jsonl"
    echo "Processed Train filename: $processed_train_file"

    # we now tokenize the file
    tokenizer_name="allenai/OLMo-2-1124-7B"
    jsonl_file="${BASE_OUTPUT_DIR}/${processed_train_file}"
    destination="${BASE_OUTPUT_DIR}/${task_prefix}-tokenized"
    echo "destination folder: $destination"

    # gzip the data if not already gzipped
    if [[ ! -f "${jsonl_file}.gz" ]]; then
      echo "Gzipping ${jsonl_file}..."
      gzip ${jsonl_file}
    else
      echo "${jsonl_file}.gz already exists. Skipping gzip."
    fi

    # tokenize the files
    dolma tokens \
      --documents ${jsonl_file}.gz \
      --tokenizer.name_or_path ${tokenizer_name} \
      --tokenizer.eos_token_id 100257 \
      --tokenizer.pad_token_id 100277 \
      --destination ${destination} \
      --dtype uint32 \
      --processes 1

    # we next add the label masks
    files=($(ls ${destination}/*.npy | grep -v mask.npy))
    echo "found these files: ${files[@]}"
    PYTHONPATH=. python -u src/scripts/eval/prepare_finetuning_masks.py \
      --token_file_paths="${files[@]}" \
      --tokenizer=${tokenizer_name} \

done