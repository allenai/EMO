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
VARIANTS=(uniform usage random)

for v in "${VARIANTS[@]}"; do
    run="${MODELS_DIR}/emo_1b14b_130b_ghost_${v}_extend${NUM_NEW_EXPERTS}_finemath_frz"
    if [ ! -d "$run" ]; then
        echo "=== skip ${v}: ${run} not found (run extend_finemath_frz.sh first) ==="
        continue
    fi
    # Latest step dir (numeric), excluding already-converted *-hf dirs.
    # Trailing `|| true`: an empty run dir (job started, no checkpoints yet) makes the `ls`
    # glob fail, and under `set -o pipefail` a bare `var=$(...)` failure trips `set -e`.
    latest=$(ls -d "${run}"/step* 2>/dev/null | grep -vE -- '-hf$' | sed 's#.*/step##' | sort -n | tail -1 || true)
    if [ -z "$latest" ]; then
        echo "=== skip ${v}: no step checkpoints in ${run} ==="
        continue
    fi
    input="${run}/step${latest}"
    output="${input}-hf"
    if [ -f "${output}/config.json" ]; then
        echo "=== skip ${v}: ${output} already exists ==="
        continue
    fi
    echo "=== converting ${v}: ${input} -> ${output} ==="
    python scripts/convert_emo_to_hf.py \
        --checkpoint-input-path "${input}" \
        --huggingface-output-dir "${output}" \
        --max-sequence-length 4096 \
        --dtype float32 \
        --validation-device cuda \
        "$@"
done
