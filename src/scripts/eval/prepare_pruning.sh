#!/bin/bash

# This script will prepare the following:
#  - Generate requests for validation set
#  - Generate requests for the train set
#  - Get expert activations for validation set

# if one step fails, the whole script fails
set -e

# parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --GROUP_NAME)
      GROUP_NAME="$2"
      shift 2
      ;;
    --BASE_OUTPUT_REMOTE_DIR)
      BASE_OUTPUT_REMOTE_DIR="$2"
      shift 2
      ;;
    --BATCH_SIZE)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --MODEL_PATH)
      MODEL_PATH="$2"
      shift 2
      ;;
    --GPUS)
      GPUS="$2"
      shift 2
      ;;
#    --)
#      VERBOSE=true
#      shift
#      ;;
    *)
      shift
      ;;
  esac
done
echo "========================================"
echo "=======Enter prepare_pruning.sh ========"
echo "GROUP_NAME: $GROUP_NAME"
echo "BASE_OUTPUT_REMOTE_DIR: $BASE_OUTPUT_REMOTE_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"

#echo "Mode: $MODE"
#[[ $VERBOSE == true ]] && echo "Verbose mode is ON"

# this prepares all model-specific variables
function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path}"
}
get_model_subpath() {
    local input="$1"
    echo "${input#*/models/}"
}
# setup output_dir to be model-specific (for validation set, as well as logits)
MODEL_NAME=$(get_model_subpath $MODEL_PATH)
model=$(get_checkpoint_name $MODEL_NAME)
output_dir="$BASE_OUTPUT_REMOTE_DIR/$model"
echo "MODEL_NAME: $MODEL_NAME"
echo "Model name for output dir: $model"
echo "Output dir: $output_dir"

# run to get requests. Will not override if file alread exists
echo "~~~~~~~~~ get validation and train examples ~~~~~~~~~"

if [[ "gsm8k" == "*$GROUP_NAME*" ]]; then
    validation_task_name="$GROUP_NAME:perplexity_validation_0shot::olmes"
    train_task_name="$GROUP_NAME:perplexity_train_0shot::olmes"
else
    validation_task_name="$GROUP_NAME:rc_validation_0shot::olmes"
    train_task_name="$GROUP_NAME:rc_train_0shot::olmes"
fi

# requests for validation (expert selection). Saves validation to model-specific directory since we also get model predictions for correctness
#PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
#      --model "$MODEL_PATH" \
#      --model-type hf \
#      --task "$validation_task_name" \
#      --output-dir $output_dir \
#      --batch-size $BATCH_SIZE \
#      --gpus $GPUS \
#      --save-raw-requests true
#
## requests for train (for finetuning). Saves to common directory since no model-specific info needed
#PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
#      --task "$train_task_name" \
#      --output-dir $BASE_OUTPUT_REMOTE_DIR \
#      --batch-size $BATCH_SIZE \
#      --save-raw-requests true
#
#echo "~~~~~~~~~ prepare expert activations on validation set ~~~~~~~~~"
#
#PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
#  --model "$MODEL_PATH" \
#  --task "$validation_task_name" \
#  --eval-dir "$output_dir" \
#  --output-dir "$output_dir" \
#  --batch-size "$BATCH_SIZE" \
#  --gpus "$GPUS" \

echo "~~~~~~~~~ tokenize the training set ~~~~~~~~~"

# this gets the correct requests and saves them into dolma format (jsonl
PYTHONPATH=. python -u src/scripts/eval/extract_finetuning_examples.py \
        --task "$train_task_name" \
        --eval-dir "$BASE_OUTPUT_REMOTE_DIR" \

get_eval_filename() {
    local task_name="$1"

    # Remove everything after and including '::' (if present)
    task_name="${task_name%%::*}"

    # Replace all ':' with '_'
    task_name="${task_name//:/_}"

    # Return the formatted string
    echo "task-${task_name}"
}

# this is the prefix of the output task name
task_prefix=$(get_eval_filename "$train_task_name")
processed_train_file="${task_prefix}-processed.jsonl"
echo "Processed Train filename: $processed_train_file"

# we now tokenize the file
tokenizer_name="allenai/OLMo-2-1124-7B"
jsonl_file="${BASE_OUTPUT_REMOTE_DIR}/${processed_train_file}"
destination="${BASE_OUTPUT_REMOTE_DIR}/${task_prefix}-tokenized"

# gzip the data
gzip ${jsonl_file}

# tokenize the files
dolma tokens \
  --documents ${jsonl_file}.gz \
  --tokenizer.name_or_path ${tokenizer_name} \
  --tokenizer.eos_token_id 100257 \
  --tokenizer.pad_token_id 100277 \
  --destination ${destination} \
  --dtype uint32 \
  --processes 1

echo "========================================"