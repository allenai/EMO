#!/usr/bin/env bash
# Migrate already-uploaded HF checkpoints to the Emo names without re-uploading
# the model weights. For each pair below, this:
#   * Renames the repo on the Hub if old != new (ModMoE_* -> Emo_*).
#   * Patches config.json's architectures / model_type / auto_map.
#   * Uploads Emo-named trust_remote_code .py files and deletes the FlexOlmo ones.
#
# Override defaults from the environment:
#   HF_ORG   target HF namespace (org or user)
#   MIGRATE  path to migrate_to_emo.py
#
# Pass extra flags through to migrate_to_emo.py via "$@", e.g.:
#   ./migrate_to_emo.sh --token "$HF_TOKEN"
#   HF_ORG=akshitab ./migrate_to_emo.sh

set -euo pipefail

HF_ORG="${HF_ORG:-allenai}"
MIGRATE="${MIGRATE:-$(dirname "${BASH_SOURCE[0]}")/migrate_to_emo.py}"

# old-name|new-name pairs (mirror upload_to_hf.sh's PAIRS).
PAIRS=(
    "ModMoE_1b14b_1T|Emo_1b14b_1T"
    "ModMoE_1b14b_130B|Emo_1b14b_130B"
    "Dense_1b_130B|Dense_1b_130B"
    "StdMoE_1b4b_130B|StdMoE_1b4b_130B"
    "StdMoE_1b14b_140B|StdMoE_1b14b_140B"
    "StdMoE_1b14b_1T|StdMoE_1b14b_1T"
)

for pair in "${PAIRS[@]}"; do
    old="${pair%%|*}"
    new="${pair##*|}"
    python "$MIGRATE" \
        --old-repo-id "${HF_ORG}/${old}" \
        --new-repo-id "${HF_ORG}/${new}" \
        "$@"
done
