#!/usr/bin/env bash
# Convert the continual-pretrained extension checkpoints (extend_finemath_frz.sh outputs)
# to HF (EmoForCausalLM) for evaluation. Auto-discovers the latest stepN dir per run so
# you don't have to know the final step. Matches the released-checkpoint conversion
# settings (--dtype float32, --max-sequence-length 4096). Idempotent.
#
#   bash scripts/models_fullextend/convert_extension_to_hf.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS_DIR="${MODELS_DIR:-models_fullextend}"
NUM_NEW_EXPERTS="${NUM_NEW_EXPERTS:-1}"

# Extension run-dir names to convert (3 ghost variants + the no-ghost baseline control).
RUNS=(
    emo_1b14b_130b_ghost_uniform_extend${NUM_NEW_EXPERTS}_finemath_frz
    emo_1b14b_130b_ghost_usage_extend${NUM_NEW_EXPERTS}_finemath_frz
    emo_1b14b_130b_ghost_random_extend${NUM_NEW_EXPERTS}_finemath_frz
    emo_1b14b_130b_noghost_extend${NUM_NEW_EXPERTS}_finemath_frz
)

for rn in "${RUNS[@]}"; do
    run="${MODELS_DIR}/${rn}"
    if [ ! -d "$run" ]; then
        echo "=== skip ${rn}: ${run} not found (run extend_finemath_frz_*.sh first) ==="
        continue
    fi
    # Latest step dir (numeric), excluding already-converted *-hf dirs.
    # Trailing `|| true`: an empty run dir (job started, no checkpoints yet) makes the `ls`
    # glob fail, and under `set -o pipefail` a bare `var=$(...)` failure trips `set -e`.
    latest=$(ls -d "${run}"/step* 2>/dev/null | grep -vE -- '-hf$' | sed 's#.*/step##' | sort -n | tail -1 || true)
    if [ -z "$latest" ]; then
        echo "=== skip ${rn}: no step checkpoints in ${run} ==="
        continue
    fi
    input="${run}/step${latest}"
    output="${input}-hf"
    if [ -f "${output}/config.json" ]; then
        echo "=== skip ${rn}: ${output} already exists ==="
        continue
    fi
    echo "=== converting ${rn}: ${input} -> ${output} ==="
    python scripts/convert_emo_to_hf.py \
        --checkpoint-input-path "${input}" \
        --huggingface-output-dir "${output}" \
        --max-sequence-length 4096 \
        --dtype float32 \
        --validation-device cuda \
        "$@"
done
