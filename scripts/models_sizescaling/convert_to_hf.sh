#!/usr/bin/env bash
# Convert models_sizescaling OLMo-core checkpoints to HF format (EmoForCausalLM).
# The driver also stages the trust_remote_code files, so the outputs load with
# stock transformers + trust_remote_code=True.
#
# Settings match the released checkpoints: --dtype float32, --max-sequence-length 4096.
# Conversion runs on CPU; validation runs on GPU (one model resident at a time).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS_DIR="${MODELS_DIR:-models_sizescaling}"

# Each entry is "<olmo-core checkpoint dir>|<HF output dir>".
# The 1b14b run predates this project and lives in the old FlexMoE tree.
MODELS=(
    "${MODELS_DIR}/emo_1b4b_130b/step30995|${MODELS_DIR}/emo_1b4b_130b/step30995-hf"
    "${MODELS_DIR}/emo_1b7b_130b/step30995|${MODELS_DIR}/emo_1b7b_130b/step30995-hf"
    "${MODELS_DIR}/emo_1b11b_130b/step30995|${MODELS_DIR}/emo_1b11b_130b/step30995-hf"
    "${HOME}/phdbrainstorm/FlexMoE/models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995|${MODELS_DIR}/emo_1b14b_130b/step30995-hf"
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
