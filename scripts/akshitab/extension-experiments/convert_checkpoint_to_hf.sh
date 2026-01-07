
MODEL_PATH=$1

echo "Converting model at ${MODEL_PATH} to Huggingface format..."

python src/examples/huggingface/convert_checkpoint_to_hf.py \
    --checkpoint-input-path "${MODEL_PATH}" \
    --max-sequence-length 4096 \
    --huggingface-output-dir "${MODEL_PATH}-hf" \
    --dtype float32 \
    --skip-validation
