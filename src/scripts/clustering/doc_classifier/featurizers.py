"""Cheap featurizers: turn documents into feature matrices for the cluster classifier.

Each featurizer produces ONE cached feature matrix for all N rows (row-aligned with the
split and labels), so it can be reused across classifiers. Featurization is *stateless*
(no train-fit here) — the train-fit normalization (tf-idf / standardize) lives in the
supervised pipeline in `classify.py`, keeping the split leakage-free.

  ngram        token-id n-grams (HashingVectorizer over token-id strings) -> sparse .npz
  token_embed  mean-pooled token embeddings from the model's own embedding matrix -> dense .npy
  oracle_router  (no featurizer needed — point classify.py at embeddings_doc_probs.npy directly)

    PYTHONPATH=.:src python -m src.scripts.clustering.doc_classifier.featurizers \
        --featurizer ngram --data-dir <cluster_dir> --docs-dir <data_dir> --out <path.npz>
"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

import numpy as np

from src.scripts.clustering.doc_classifier.data import build_row_map, stream_token_ids

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def featurize_ngram(data_dir, docs_dir, out, token_cap=1024, ngram=(1, 2), log2_features=20):
    """Token-id n-grams via stateless hashing -> CSR (N, 2**log2_features)."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer

    row_map, n = build_row_map(data_dir)
    hv = HashingVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        ngram_range=tuple(ngram),
        n_features=2**log2_features,
        alternate_sign=False,
        norm=None,
    )
    parts, row_ids = [], []
    for ids, toks in stream_token_ids(docs_dir, row_map, token_cap):
        parts.append(hv.transform(" ".join(map(str, t)) for t in toks))
        row_ids.extend(ids)
        logger.info(f"  featurized {len(row_ids):,}/{n:,}")
    row_ids = np.asarray(row_ids)
    assert (
        len(row_ids) == n and len(np.unique(row_ids)) == n
    ), f"coverage mismatch: {len(row_ids)} streamed, {len(np.unique(row_ids))} unique, N={n}"
    X = sp.vstack(parts).tocsr()[np.argsort(row_ids)]  # reorder to row order 0..N-1
    sp.save_npz(out, X)
    logger.info(f"saved {out}: shape={X.shape}, nnz={X.nnz:,} ({X.nnz / n:.0f}/row)")


def _load_embed_matrix(model_path: str) -> np.ndarray:
    """Load only embed_tokens.weight (vocab, hidden) from an HF checkpoint, as float32."""
    import glob
    import json as _json

    from safetensors import safe_open

    def _from(fp, key):
        with safe_open(fp, framework="pt") as f:
            return f.get_tensor(key).float().numpy()

    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = _json.load(open(idx))["weight_map"]
        for k, shard in wm.items():
            if k.endswith("embed_tokens.weight"):
                return _from(os.path.join(model_path, shard), k)
    for fp in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    return f.get_tensor(k).float().numpy()
    raise KeyError(f"embed_tokens.weight not found in {model_path}")


def featurize_token_embed(data_dir, docs_dir, model_path, out, token_cap=1024):
    """Mean-pooled token embeddings -> dense (N, hidden) float32."""
    W = _load_embed_matrix(model_path)
    logger.info(f"embed matrix {W.shape} from {model_path}")
    row_map, n = build_row_map(data_dir)
    X = np.zeros((n, W.shape[1]), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    done = 0
    for ids, toks in stream_token_ids(docs_dir, row_map, token_cap):
        for i, t in zip(ids, toks):
            if t:
                X[i] = W[np.asarray(t, dtype=np.int64)].mean(0)
            seen[i] = True
        done += len(ids)
        logger.info(f"  pooled {done:,}/{n:,}")
    assert seen.all(), f"{(~seen).sum()} rows never streamed"
    np.save(out, X)
    logger.info(f"saved {out}: shape={X.shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--featurizer", required=True, choices=["ngram", "token_embed"])
    ap.add_argument("--data-dir", required=True, help="cluster dir with doc_ids.npz")
    ap.add_argument("--docs-dir", required=True, help="data dir with docs-*.jsonl.gz")
    ap.add_argument(
        "--out", required=True, help="output path (.npz for ngram, .npy for token_embed)"
    )
    ap.add_argument("--token-cap", type=int, default=1024)
    ap.add_argument("--ngram", type=int, nargs=2, default=(1, 2))
    ap.add_argument("--log2-features", type=int, default=20)
    ap.add_argument("--model-path", default="models_v2/emo_64exp_50b_wsd_lr2e-3/step23842-hf")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.featurizer == "ngram":
        featurize_ngram(
            args.data_dir, args.docs_dir, args.out, args.token_cap, args.ngram, args.log2_features
        )
    else:
        featurize_token_embed(
            args.data_dir, args.docs_dir, args.model_path, args.out, args.token_cap
        )


if __name__ == "__main__":
    main()
