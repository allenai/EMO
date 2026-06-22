#!/usr/bin/env bash
# Build the full-size router-fixed step-0 init checkpoint for models_routerfixed.
#
# One-time prep (run on a GPU-attached / big-RAM session; the build itself is CPU-only and
# single-process -- no torchrun). Grafts the trained step-11921 routers onto a fresh EMO init
# and writes a model-only checkpoint that the two training runs load via --load_path.
#
#   bash scripts/models_routerfixed/build_step0.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$(pwd)/src"

EXPERIMENT_DIR="models_routerfixed"
SRC_RUN="emo_1b14b_50bof130b"                              # the trained baseline (models_fullextend)
SRC_STEP="step11921"
OUT_NAME="init_routerfixed_step0"

# Provenance symlink: models_routerfixed/emo_1b14b_50bof130b -> models_fullextend/emo_1b14b_50bof130b.
# Gives the builder a stable read path inside the experiment dir and documents the source.
mkdir -p "${EXPERIMENT_DIR}"
if [[ ! -e "${EXPERIMENT_DIR}/${SRC_RUN}" ]]; then
    ln -s "$(pwd)/models_fullextend/${SRC_RUN}" "${EXPERIMENT_DIR}/${SRC_RUN}"
    echo "Created symlink ${EXPERIMENT_DIR}/${SRC_RUN} -> models_fullextend/${SRC_RUN}"
fi

python scripts/models_routerfixed/build_step0_routerfixed.py \
    --src-checkpoint "${EXPERIMENT_DIR}/${SRC_RUN}/${SRC_STEP}" \
    --out-dir "${EXPERIMENT_DIR}/${OUT_NAME}"

echo "Done. Init checkpoint at ${EXPERIMENT_DIR}/${OUT_NAME}/model_and_optim"
