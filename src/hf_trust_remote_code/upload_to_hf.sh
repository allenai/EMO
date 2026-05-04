#!/usr/bin/env bash
# Upload all Emo checkpoints to the Hugging Face Hub with the
# trust_remote_code .py files staged in. Repos are private by default.
#
# Override defaults from the environment:
#   MODEL_ROOT  root holding the local *-hf checkpoint dirs
#   HF_ORG      target HF namespace (org or user)
#   UPLOAD      path to upload_to_hf.py
#
# Pass extra flags through to upload_to_hf.py via "$@", e.g.:
#   ./upload_all.sh --no-upload      # dry-run all six
#   ./upload_all.sh --public         # push as public
#   HF_ORG=akshitab ./upload_all.sh

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/weka/oe-training-default/ryanwang/phdbrainstorm/Emo/models}"
HF_ORG="${HF_ORG:-allenai}"
UPLOAD="${UPLOAD:-$(dirname "${BASH_SOURCE[0]}")/upload_to_hf.py}"

# Local-relative-path  HF-repo-name pairs.
PAIRS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf|Emo_1b14b_1T"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf|Emo_1b14b_130B"
    "dense_1b_lr-4e-3_0213/step30995-hf|Dense_1b_130B"
    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308/step30995-hf|StdMoE_1b4b_130B"
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf|StdMoE_1b14b_140B"
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339-hf|StdMoE_1b14b_1T"
)

for pair in "${PAIRS[@]}"; do
    rel="${pair%%|*}"
    name="${pair##*|}"
    echo "==> ${name}"
    python "$UPLOAD" \
        --model-path "${MODEL_ROOT}/${rel}" \
        --repo-id    "${HF_ORG}/${name}" \
        "$@"
done

