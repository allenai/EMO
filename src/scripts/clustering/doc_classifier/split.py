"""Build the shared train/val/test split over document row indices.

Every Stage-2 step (oracle clustering, all featurizers, all classifiers) reads this one
`split.npz` so methods are compared on identical documents. The split is over row indices
`0..N-1` (aligned with `embeddings_*.npy`, `assignments.npy`, `doc_ids.npz`, and line `i` of
`metadata_docs.jsonl.gz`), **stratified by `source`** (the coarse data-mix label in the
metadata, available before any clustering) so no split over-/under-represents a data source.

    PYTHONPATH=.:src python -m src.scripts.clustering.doc_classifier.split \
        --data-dir modular_extension/cluster/emo100b_step23842
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sources(data_dir: str) -> np.ndarray:
    """Per-row `source` label from metadata_docs.jsonl.gz (row i -> document i)."""
    path = os.path.join(data_dir, "metadata_docs.jsonl.gz")
    sources = []
    with gzip.open(path, "rt") as f:
        for line in f:
            sources.append(json.loads(line)["source"])
    return np.asarray(sources)


def stratified_split(
    sources: np.ndarray, fracs: tuple[float, float, float], seed: int
) -> dict[str, np.ndarray]:
    """Split row indices into train/val/test, stratified within each source."""
    ftr, fva, _ = fracs
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for s in np.unique(sources):
        idx = np.where(sources == s)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * ftr))
        n_va = int(round(n * fva))
        train.append(idx[:n_tr])
        val.append(idx[n_tr : n_tr + n_va])
        test.append(idx[n_tr + n_va :])
    out = {
        "train_idx": np.sort(np.concatenate(train)).astype(np.int64),
        "val_idx": np.sort(np.concatenate(val)).astype(np.int64),
        "test_idx": np.sort(np.concatenate(test)).astype(np.int64),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Dir with metadata_docs.jsonl.gz")
    ap.add_argument(
        "--out", default=None, help="Output .npz (default: <data-dir>/doc_classifier/split.npz)"
    )
    ap.add_argument(
        "--fracs", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TR", "VA", "TE")
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(sum(args.fracs) - 1.0) < 1e-6, f"fracs must sum to 1, got {args.fracs}"
    out = args.out or os.path.join(args.data_dir, "doc_classifier", "split.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    sources = load_sources(args.data_dir)
    n = len(sources)
    logger.info(
        f"{n:,} docs, {len(np.unique(sources))} sources; fracs={args.fracs} seed={args.seed}"
    )

    split = stratified_split(sources, tuple(args.fracs), args.seed)
    sizes = {k: len(v) for k, v in split.items()}
    assert sum(sizes.values()) == n, f"split loses rows: {sizes} vs {n}"
    # disjoint + full coverage
    allidx = np.concatenate([split["train_idx"], split["val_idx"], split["test_idx"]])
    assert len(np.unique(allidx)) == n, "splits overlap or miss rows"

    np.savez(
        out,
        train_idx=split["train_idx"],
        val_idx=split["val_idx"],
        test_idx=split["test_idx"],
        fracs=np.asarray(args.fracs, dtype=np.float64),
        seed=np.int64(args.seed),
    )
    logger.info(
        f"saved {out}: train={sizes['train_idx']:,} val={sizes['val_idx']:,} test={sizes['test_idx']:,}"
    )
    # report source-proportion preservation (max abs deviation across sources)
    for name, key in [("train", "train_idx"), ("val", "val_idx"), ("test", "test_idx")]:
        sub = sources[split[key]]
        dev = max(abs((sub == s).mean() - (sources == s).mean()) for s in np.unique(sources))
        logger.info(f"  {name}: max source-proportion deviation from global = {dev:.4f}")


if __name__ == "__main__":
    main()
