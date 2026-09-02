#!/bin/bash
# Stage 2 Part B: featurize documents cheaply and train a classifier to predict their cluster.
# Ground truth = the train-fit clustering labels (from cluster_oracle_split.sh). CPU, local.
#
#   FEATURIZER=ngram        bash scripts/modular_extension/classify_clusters.sh
#   FEATURIZER=token_embed  bash scripts/modular_extension/classify_clusters.sh
#   FEATURIZER=oracle_router bash scripts/modular_extension/classify_clusters.sh   # ceiling
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=".:src"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"

DATA_DIR="${DATA_DIR:-modular_extension/cluster/emo100b_step23842}"
DOCS_DIR="${DOCS_DIR:-modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-110B}"
OUT_ROOT="${OUT_ROOT:-$DATA_DIR/doc_classifier}"
FEATURIZER="${FEATURIZER:-ngram}"
CLASSIFIER="${CLASSIFIER:-logreg}"
TOKEN_CAP="${TOKEN_CAP:-1024}"

SPLIT="$OUT_ROOT/split.npz"
LABELS="$OUT_ROOT/oracle/train_model_labels.npy"
FEATDIR="$OUT_ROOT/features"
mkdir -p "$FEATDIR"

FEAT() { python -m src.scripts.clustering.doc_classifier.featurizers "$@"; }

case "$FEATURIZER" in
  ngram)
    FEATS="$FEATDIR/ngram_cap${TOKEN_CAP}.npz"
    [ -f "$FEATS" ] || FEAT --featurizer ngram --data-dir "$DATA_DIR" --docs-dir "$DOCS_DIR" \
        --out "$FEATS" --token-cap "$TOKEN_CAP"
    ;;
  token_embed)
    FEATS="$FEATDIR/token_embed_cap${TOKEN_CAP}.npy"
    [ -f "$FEATS" ] || FEAT --featurizer token_embed --data-dir "$DATA_DIR" --docs-dir "$DOCS_DIR" \
        --out "$FEATS" --token-cap "$TOKEN_CAP"
    ;;
  oracle_router)
    FEATS="$DATA_DIR/embeddings_doc_probs.npy"   # the fingerprint we clustered on = ceiling
    ;;
  *)
    echo "unknown FEATURIZER=$FEATURIZER (ngram|token_embed|oracle_router)"; exit 1;;
esac

python -m src.scripts.clustering.doc_classifier.classify \
    --features "$FEATS" --split "$SPLIT" --labels "$LABELS" \
    --out "$OUT_ROOT/${FEATURIZER}_${CLASSIFIER}" --classifier "$CLASSIFIER"
