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
    --BASE_OUTPUT_DIR)
      BASE_OUTPUT_DIR="$2"
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
    *)
      shift
      ;;
  esac
done
echo "========================================"
echo "=======Enter prepare_pruning.sh ========"
echo "GROUP_NAME: $GROUP_NAME"
echo "BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"

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
output_dir="$BASE_OUTPUT_DIR/$model"
echo "MODEL_NAME: $MODEL_NAME"
echo "Model name for output dir: $model"
echo "Output dir: $output_dir"

# run to get requests. Will not override if file alread exists
echo "~~~~~~~~~ get validation and train examples ~~~~~~~~~"

if [[ "$GROUP_NAME" == "*gsm8k*" ]]; then
    validation_task_name="$GROUP_NAME:perplexity_validation_0shot::olmes"
    train_task_name="$GROUP_NAME:perplexity_train_0shot::olmes"
else
    validation_task_name="$GROUP_NAME:rc_validation_0shot::olmes"
    train_task_name="$GROUP_NAME:rc_train_0shot::olmes"
fi

# requests for validation (expert selection). Saves validation to model-specific directory since we also get model predictions for correctness
PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
      --model "$MODEL_PATH" \
      --model-type hf \
      --task "$validation_task_name" \
      --output-dir $output_dir \
      --batch-size $BATCH_SIZE \
      --gpus $GPUS \
      --save-raw-requests true

# requests for train (for finetuning). Saves to common directory since no model-specific info needed
PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
      --task "$train_task_name" \
      --output-dir $BASE_OUTPUT_DIR \
      --batch-size $BATCH_SIZE \
      --save-raw-requests true

echo "~~~~~~~~~ prepare expert activations on validation set ~~~~~~~~~"

PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
  --model "$MODEL_PATH" \
  --task "$validation_task_name" \
  --eval-dir "$output_dir" \
  --output-dir "$output_dir" \
  --batch-size "$BATCH_SIZE" \
  --gpus "$GPUS" \

echo "~~~~~~~~~ prepare tokenization of the training set ~~~~~~~~~"

# this gets the correct requests and saves them into dolma format (jsonl
PYTHONPATH=. python -u src/scripts/eval/extract_finetuning_examples.py \
        --task "$train_task_name" \
        --eval-dir "$BASE_OUTPUT_DIR" \


echo "========================================"