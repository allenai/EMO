PATH_TO_MODEL_DIR=""

MODELS=(
    "model1/step2385"
)

for MODEL in "${MODELS[@]}"; do
#
    python src/examples/huggingface/convert_checkpoint_to_hf.py \
          --checkpoint-input-path "${PATH_TO_MODEL_DIR}/$MODEL" \
          --max-sequence-length 4096 \
          --huggingface-output-dir "${PATH_TO_MODEL_DIR}/$MODEL-hf" \
          --dtype float32 \
          --skip-validation

done
