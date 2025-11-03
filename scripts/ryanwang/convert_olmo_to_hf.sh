BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"
MODELS=(
  "dense_1b_olmoe-mix_300B_1030"
  "moe_1b7b_olmoe-mix_300B_1030"
)

for MODEL in "${MODELS[@]}"; do
  # list all the files in the model directory, only include directories that start with "step"
  MODEL_DIR="${BASE_FOLDER}/${MODEL}"
  STEPS=$(ls -d ${MODEL_DIR}/step*/ | xargs -n 1 basename)
  for STEP in ${STEPS}; do
    # check that the hf converted folder does not already exist
    if [ -d "${MODEL_DIR}/${STEP}-hf" ]; then
      break 1
    fi
    echo "Converting model ${MODEL} at step ${STEP} to Huggingface format..."

    python src/examples/huggingface/convert_checkpoint_to_hf.py \
      --checkpoint-input-path "${MODEL_DIR}/${STEP}" \
      --max-sequence-length 4096 \
      --huggingface-output-dir "${MODEL_DIR}/${STEP}-hf" \
      --dtype float32

  done
done

