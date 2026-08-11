"""
Subsample-stability experiment: how much of the full-data oracle partition does a
clustering fit on a subsample recover?

Every arm ends the same way — assign ALL rows of the data dir to the arm's fitted
centroids and save the labels — so arms are directly comparable to the reference
full-fit partition (``assignments.npy`` from cluster.py --save).

Modes (``run`` subcommand; one arm per invocation, idempotent):
  honest    sample n rows from the RAW embedding; fit mean/PCA(95% var)/L2 AND
            spherical k-means on the subsample only (nothing borrowed from the
            full fit); assign all rows through the subsample-fit transform.
  frozen    identical row subsample (same seed => same rows), but reuse the full
            fit's preprocessed cache (``preprocessed_<emb>_<pre>.npy``); only
            k-means is re-fit. The honest-vs-frozen gap isolates transform
            instability from k-means instability.
  fullseed  re-fit k-means on the FULL preprocessed cache with a different seed
            (the reference used seed 42) — the seed-noise ceiling.

``score`` compares every arm's labels against the reference assignments:
Hungarian-matched accuracy (doc-level and doc_len-token-weighted), ARI, NMI.

Usage:
    PYTHONPATH=.:src python -m src.scripts.clustering.subsample_stability run \\
        --data-dir modular_extension/cluster/emo100b_step23842_100B-130B \\
        --mode honest --n-sample 1000000 --seed 0

    PYTHONPATH=.:src python -m src.scripts.clustering.subsample_stability score \\
        --data-dir modular_extension/cluster/emo100b_step23842_100B-130B \\
        --reference-dir .../doc_probs_mean_pca_l2_spherical_kmeans_k64
"""

import argparse
import json
import logging
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

from src.scripts.clustering.cluster import fit_spherical_kmeans
from src.scripts.clustering.transform import (
    EMBEDDING_FILES,
    apply_mean_pca_l2,
    fit_apply_mean_pca_l2,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def arm_name(mode: str, n_sample: int | None, seed: int) -> str:
    if mode == "fullseed":
        return f"fullseed_seed{seed}"
    return f"{mode}_n{n_sample}_seed{seed}"


def predict_chunked(km, x: np.ndarray, chunk: int, transform_state=None) -> np.ndarray:
    """Assign every row of ``x`` (mmap ok) to km's nearest L2 centroid, chunkwise.

    With ``transform_state`` the raw rows are pushed through the fitted
    mean/PCA/L2 first (honest mode); otherwise rows are assumed to already be in
    the preprocessed (L2-normalized) space.
    """
    from sklearn.preprocessing import normalize

    n = x.shape[0]
    labels = np.empty(n, dtype=np.int16)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        block = np.asarray(x[lo:hi], dtype=np.float32)
        if transform_state is not None:
            block = apply_mean_pca_l2(block, transform_state)
        else:
            block = normalize(block, norm="l2")
        labels[lo:hi] = km.predict(block)
        logger.info(f"  assigned {hi:,}/{n:,} rows ({time.time() - t0:.0f}s)")
    return labels


def run(args):
    out_dir = os.path.join(args.out_root, arm_name(args.mode, args.n_sample, args.seed))
    labels_path = os.path.join(out_dir, "labels.npy")
    if os.path.exists(labels_path):
        logger.info(f"SKIP: {labels_path} already exists")
        return
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(args.data_dir, EMBEDDING_FILES[args.embedding])
    cache_path = os.path.join(args.data_dir, args.preprocess_cache)
    t_start = time.time()
    run_info: dict = {
        "mode": args.mode,
        "n_sample": args.n_sample,
        "seed": args.seed,
        "k": args.k,
        "embedding": args.embedding,
    }

    if args.mode == "honest":
        raw = np.load(raw_path, mmap_mode="r")
        n_total = raw.shape[0]
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, args.n_sample, replace=False))
        logger.info(f"honest: gathering {args.n_sample:,}/{n_total:,} raw rows")
        sub = np.asarray(raw[idx], dtype=np.float32)
        transformed_sub, state = fit_apply_mean_pca_l2(sub)
        del sub
        run_info["n_components"] = int(state[1].n_components_)
        logger.info(f"honest: subsample PCA kept {run_info['n_components']} components")
        km = fit_spherical_kmeans(transformed_sub, args.k, random_state=args.seed)
        del transformed_sub
        labels = predict_chunked(km, raw, args.chunk_size, transform_state=state)

    elif args.mode == "frozen":
        pre = np.load(cache_path, mmap_mode="r")
        n_total = pre.shape[0]
        rng = np.random.default_rng(args.seed)  # same draw as honest at equal seed
        idx = np.sort(rng.choice(n_total, args.n_sample, replace=False))
        logger.info(f"frozen: gathering {args.n_sample:,}/{n_total:,} preprocessed rows")
        sub = np.asarray(pre[idx], dtype=np.float32)
        run_info["n_components"] = int(pre.shape[1])
        km = fit_spherical_kmeans(sub, args.k, random_state=args.seed)
        del sub
        labels = predict_chunked(km, pre, args.chunk_size)

    elif args.mode == "fullseed":
        logger.info(f"fullseed: loading full preprocessed cache {cache_path}")
        pre = np.load(cache_path)
        run_info["n_components"] = int(pre.shape[1])
        km = fit_spherical_kmeans(pre, args.k, random_state=args.seed)
        labels = predict_chunked(km, pre, args.chunk_size)

    else:
        raise ValueError(f"unknown mode {args.mode}")

    sizes = np.bincount(labels, minlength=args.k)
    run_info["cluster_size_min"] = int(sizes.min())
    run_info["cluster_size_median"] = int(np.median(sizes))
    run_info["cluster_size_max"] = int(sizes.max())
    run_info["n_empty_clusters"] = int((sizes == 0).sum())
    run_info["n_rows_assigned"] = int(labels.shape[0])
    run_info["runtime_s"] = round(time.time() - t_start, 1)

    np.save(labels_path, labels)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    logger.info(f"DONE {arm_name(args.mode, args.n_sample, args.seed)}: {run_info}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def contingency(ref: np.ndarray, lab: np.ndarray, k: int, weights=None) -> np.ndarray:
    """(k, k) contingency of reference rows vs arm columns, optionally weighted."""
    flat = ref.astype(np.int64) * k + lab.astype(np.int64)
    c = np.bincount(flat, weights=weights, minlength=k * k).reshape(k, k)
    return c.astype(np.float64)


def hungarian_accuracy(c: np.ndarray) -> float:
    """Best-case one-to-one relabeling agreement from a contingency matrix."""
    from scipy.optimize import linear_sum_assignment

    r, cc = linear_sum_assignment(-c)
    return float(c[r, cc].sum() / c.sum())


def ari_from_contingency(c: np.ndarray) -> float:
    def comb2(x):
        return x * (x - 1) / 2.0

    sum_cells = comb2(c).sum()
    a = comb2(c.sum(axis=1)).sum()
    b = comb2(c.sum(axis=0)).sum()
    n = c.sum()
    expected = a * b / comb2(n)
    max_index = (a + b) / 2.0
    return float((sum_cells - expected) / (max_index - expected))


def nmi_from_contingency(c: np.ndarray) -> float:
    n = c.sum()
    pij = c / n
    pi = pij.sum(axis=1)
    pj = pij.sum(axis=0)
    nz = pij > 0
    mi = (pij[nz] * np.log(pij[nz] / np.outer(pi, pj)[nz])).sum()
    hi = -(pi[pi > 0] * np.log(pi[pi > 0])).sum()
    hj = -(pj[pj > 0] * np.log(pj[pj > 0])).sum()
    return float(mi / np.sqrt(hi * hj))


def score(args):
    ref = np.load(os.path.join(args.reference_dir, "assignments.npy"))
    k = int(ref.max()) + 1
    doc_len = np.load(os.path.join(args.data_dir, "doc_ids.npz"))["doc_len"].astype(np.float64)
    assert doc_len.shape[0] == ref.shape[0]

    results = {}
    for name in sorted(os.listdir(args.out_root)):
        labels_path = os.path.join(args.out_root, name, "labels.npy")
        if not os.path.exists(labels_path):
            continue
        lab = np.load(labels_path)
        assert lab.shape[0] == ref.shape[0], f"{name}: row count mismatch"
        c_doc = contingency(ref, lab, k)
        c_tok = contingency(ref, lab, k, weights=doc_len)
        with open(os.path.join(args.out_root, name, "run.json")) as f:
            run_info = json.load(f)
        results[name] = {
            **run_info,
            "acc_docs": round(hungarian_accuracy(c_doc), 4),
            "acc_tokens": round(hungarian_accuracy(c_tok), 4),
            "ari": round(ari_from_contingency(c_doc), 4),
            "nmi": round(nmi_from_contingency(c_doc), 4),
        }
        logger.info(f"{name}: {results[name]}")

    out_path = os.path.join(args.out_root, "scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path} ({len(results)} arms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run one arm")
    pr.add_argument("--data-dir", required=True)
    pr.add_argument("--mode", required=True, choices=["honest", "frozen", "fullseed"])
    pr.add_argument("--n-sample", type=int, default=None, help="rows to fit on (honest/frozen)")
    pr.add_argument("--seed", type=int, required=True, help="subsample draw AND k-means seed")
    pr.add_argument("--k", type=int, default=64)
    pr.add_argument("--embedding", default="doc_probs")
    pr.add_argument("--preprocess-cache", default="preprocessed_doc_probs_mean_pca_l2.npy")
    pr.add_argument("--out-root", default=None)
    pr.add_argument("--chunk-size", type=int, default=1_000_000)

    ps = sub.add_parser("score", help="score all arms against the reference partition")
    ps.add_argument("--data-dir", required=True)
    ps.add_argument("--reference-dir", required=True)
    ps.add_argument("--out-root", default=None)

    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = os.path.join(args.data_dir, "subsample_stability")
    if args.cmd == "run":
        if args.mode in ("honest", "frozen") and not args.n_sample:
            parser.error("--n-sample required for honest/frozen")
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
