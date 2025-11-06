#!/bin/bash

# This script will prepare the following:
#  - Generate requests for validation set
#  - Generate requests for the train set
#  - Tokenize the train set
#  - Get expert activations for validation set

# parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --TASK)
      TASK="$2"
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
echo "TASK: $TASK"
echo "BASE_OUTPUT_REMOTE_DIR: $BASE_OUTPUT_REMOTE_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"

#echo "Mode: $MODE"
#[[ $VERBOSE == true ]] && echo "Verbose mode is ON"

PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
        --task $TASK \
        --output-dir $BASE_OUTPUT_REMOTE_DIR \
        --batch-size $BATCH_SIZE \
        --save-raw-requests \

echo "========================================"