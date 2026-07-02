#!/usr/bin/env bash
# Convert the EMO 64e WSD-2e-3 trunk's 100B-token checkpoint (step23842, exactly 100.0B tokens)
# to an HF trust_remote_code checkpoint for the document-level router-clustering pipeline
# (scripts/modular_extension/launch_embed_docs.sh). Same flags as the released-checkpoint /
# merged-eval conversions (scripts/models_v2/launch_merged_eval.sh:ensure_hf): fp32 weights,
# max seq len 4096, conversion on CPU with GPU logit validation. Idempotent.
#
#   bash scripts/modular_extension/convert_100b_to_hf.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN=models_v2/emo_64exp_50b_wsd_lr2e-3
STEP=23842
SRC="${RUN}/step${STEP}"
DST="${RUN}/step${STEP}-hf"

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
