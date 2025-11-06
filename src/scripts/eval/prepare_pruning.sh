#!/bin/bash

# This script will prepare the following:
#  - Generate requests for validation set
#  - Generate requests for the train set
#  - Get expert activations for validation set

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


# run to get requests. Will not override if file alread exists
echo "~~~~~~~~~ get validation and train examples ~~~~~~~~~"

if [[ "gsm8k" == "*$GROUP_NAME*" ]]; then
    validation_task_name="$GROUP_NAME:perplexity_validation_0shot::olmes"
    train_task_name="$GROUP_NAME:perplexity_train_0shot::olmes"
else
    validation_task_name="$GROUP_NAME:rc_validation_0shot::olmes"
    train_task_name="$GROUP_NAME:rc_train_0shot::olmes"
fi

# requests for validation (expert selection)
PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
      --task "$validation_task_name" \
      --output-dir $BASE_OUTPUT_REMOTE_DIR \
      --batch-size $BATCH_SIZE \
      --save-raw-requests true
# requests for train (for finetuning)
PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
      --task "$train_task_name" \
      --output-dir $BASE_OUTPUT_REMOTE_DIR \
      --batch-size $BATCH_SIZE \
      --save-raw-requests true

echo "~~~~~~~~~ prepare expert activations on validation set ~~~~~~~~~"

function get_checkpoint_name {
    local path=$1
    local split_path=${path#*OLMo2-7B-}
    local modified_path=${split_path//\//_}
    modified_path=$(echo $modified_path | sed 's/^_//;s/_$//')
    echo "${modified_path//hf/${model_type}}"
}
model=$(get_checkpoint_name $MODEL_PATH)
output_dir="$BASE_OUTPUT_REMOTE_DIR/$model"

PYTHONPATH=. python -u src/scripts/eval/launch_logits.py \
  --model "$MODEL_PATH" \
  --task "$validation_task_name" \
  --eval-dir "$BASE_OUTPUT_REMOTE_DIR" \
  --output-dir "$output_dir" \
  --batch-size "$BATCH_SIZE" \
  --gpus "$GPUS" \
  --use_correct_only

echo "========================================"