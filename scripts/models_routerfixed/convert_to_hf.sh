#!/usr/bin/env bash
# Convert the models_routerfixed noaux OLMo-core checkpoint to HF (EmoForCausalLM) so the
# expert token-matching / clustering analysis (which runs on HF checkpoints) can consume it.
#
# The baseline (models_fullextend/emo_1b14b_50bof130b/step11921) is already converted at
# .../step11921-hf, so only the noaux run needs converting here. Settings match the released
# checkpoints / sizescaling: --dtype float32, --max-sequence-length 4096. Conversion runs on CPU;
# logit-validation runs on GPU. Idempotent: skips if the HF config.json already exists.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=(
    "models_routerfixed/emo_1b14b_50bof130b_routerfixed_noaux/step11921|models_routerfixed/emo_1b14b_50bof130b_routerfixed_noaux/step11921-hf"
)

for PAIR in "${MODELS[@]}"; do
    INPUT="${PAIR%%|*}"
    OUTPUT="${PAIR##*|}"
    if [ -f "${OUTPUT}/config.json" ]; then
        echo "=== skipping ${INPUT} (${OUTPUT} already exists) ==="
        continue
    fi
    echo "=== converting ${INPUT} -> ${OUTPUT} ==="
    python scripts/convert_emo_to_hf.py \
        --checkpoint-input-path "${INPUT}" \
        --huggingface-output-dir "${OUTPUT}" \
        --max-sequence-length 4096 \
        --dtype float32 \
        --validation-device cuda \
        "$@"
done

echo "=== Done. ==="
