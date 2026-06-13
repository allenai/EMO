#!/usr/bin/env bash
# Convert models_fullextend ghost checkpoints to HF (EmoForCausalLM), staging the
# (ghost-enabled) trust_remote_code files so they load with trust_remote_code=True.
# Settings match the released checkpoints: --dtype float32, --max-sequence-length 4096.
# Conversion runs on CPU; validation (logit match vs OLMo-core, ghost OFF) on GPU.
# Idempotent: skips outputs that already exist.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS_DIR="${MODELS_DIR:-models_fullextend}"

# "<olmo-core checkpoint dir>|<HF output dir>". Add configs as they finish.
MODELS=(
    "${MODELS_DIR}/emo_1b14b_130b_ghost_usage_always_detachF/step11921|${MODELS_DIR}/emo_1b14b_130b_ghost_usage_always_detachF/step11921-hf"
    "${MODELS_DIR}/emo_1b14b_130b_ghost_uniform_always_detachF/step11921|${MODELS_DIR}/emo_1b14b_130b_ghost_uniform_always_detachF/step11921-hf"
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

# Materialize the ghost-on variant (symlinked weights + config.json with the toggle) for
# each converted checkpoint, so the MC9 eval can run the with-ghost arm. The eval-time
# ghost coeff_mode is matched to the run's training coeff_mode, read off the runname
# (..._ghost_<mode>_always_detachF/...): usage -> usage, uniform -> uniform, random -> random.
for PAIR in "${MODELS[@]}"; do
    OUTPUT="${PAIR##*|}"
    if [ -f "${OUTPUT}/config.json" ]; then
        case "$OUTPUT" in
            *_ghost_uniform_*) COEFF_MODE=uniform ;;
            *_ghost_random_*)  COEFF_MODE=random ;;
            *)                 COEFF_MODE=usage ;;
        esac
        python scripts/models_fullextend/make_ghost_hf_variant.py --src "${OUTPUT}" --coeff-mode "${COEFF_MODE}"
    fi
done
