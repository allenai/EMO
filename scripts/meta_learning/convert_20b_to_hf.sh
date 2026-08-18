#!/usr/bin/env bash
# PARENT: "scripts/modular_extension/convert_100b_to_hf.sh"
# DESCRIPTION:
#     Convert a meta_learning 128e arm's 20B-token checkpoint (step4768) to an HF
#     trust_remote_code checkpoint for the phase-2 document-level router-clustering
#     pipeline (scripts/meta_learning/launch_embed_docs.sh). Same flags as the
#     released-checkpoint conversions: fp32 weights, max seq len 4096, conversion on
#     CPU with GPU logit validation. Idempotent.
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/convert_20b_to_hf.sh
#   MODEL=vanilla          bash scripts/meta_learning/convert_20b_to_hf.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
STEP="${STEP:-4768}"
SRC="meta_learning/meta128_${MODEL}/step${STEP}"
DST="meta_learning/meta128_${MODEL}/step${STEP}-hf"

if [ -f "${DST}/config.json" ]; then
    echo "HF checkpoint already exists at ${DST} -- skipping conversion."
    exit 0
fi
[ -d "${SRC}" ] || { echo "ERROR: source checkpoint ${SRC} missing" >&2; exit 1; }

python scripts/convert_emo_to_hf.py \
    --checkpoint-input-path "${SRC}" \
    --huggingface-output-dir "${DST}" \
    --max-sequence-length 4096 \
    --dtype float32 \
    --validation-device cuda
