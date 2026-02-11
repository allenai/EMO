MODEL_PATH="${1:-/data/input/kevinf/checkpoints/olmo3-1b-10B-chempile-papers_education_lift-/step2385}"
MODEL_PATH="${MODEL_PATH%/}"
OUTPUT_PATH="${2:-${MODEL_PATH}-hf}"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i $MODEL_PATH \
    -o $OUTPUT_PATH \
    --skip-validation \
    --max-sequence-length 65536