#!/bin/bash
# Stage 2 Part A: build the shared split, fit the train-only + global clusterings (one shared
# implementation), and report the train-vs-global agreement gate. CPU, local, on existing
# router embeddings. See src/scripts/clustering/doc_classifier/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=".:src"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"

DATA_DIR="${DATA_DIR:-modular_extension/cluster/emo100b_step23842}"
K="${K:-64}"

python -m src.scripts.clustering.doc_classifier.split --data-dir "$DATA_DIR"
python -m src.scripts.clustering.doc_classifier.cluster_oracle --data-dir "$DATA_DIR" --k "$K"
