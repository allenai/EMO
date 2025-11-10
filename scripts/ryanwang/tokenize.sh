#!/bin/bash

BASE_OUTPUT_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/prune"

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
#  "arc_easy:rc_train_0shot::olmes"
#  "arc_challenge:rc_train_0shot::olmes"
#  "boolq:rc_train_0shot::olmes"
#  "csqa:rc_train_0shot::olmes"
  "hellaswag:rc_train_0shot::olmes"
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
#    dolma tokens \
#      --documents ${jsonl_file}.gz \
#      --tokenizer.name_or_path ${tokenizer_name} \
#      --tokenizer.eos_token_id 100257 \
#      --tokenizer.pad_token_id 100277 \
#      --destination ${destination} \
#      --dtype uint32 \
#      --processes 1

    # we next add the label masks
    files=($(ls ${destination}/*.npy | grep -v mask.npy))
    echo "found these files: ${files[@]}"
    PYTHONPATH=. python -u src/scripts/eval/prepare_finetuning_masks.py \
      --token_file_paths="${files[@]}" \
      --tokenizer=${tokenizer_name} \

done