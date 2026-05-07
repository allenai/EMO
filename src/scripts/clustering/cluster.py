"""
Cluster router embeddings and evaluate quality.

Loads a derived embedding (from transform.py), applies preprocessing,
clusters, evaluates, and optionally saves results.

Usage:
    # Sweep over k values
    python -m src.scripts.clustering.cluster \\
        --data-dir .../pretraining/<model>/ \\
        --embedding probs --preprocess mean_pca_l2 \\
        --method spherical_kmeans --k 16 32 64 128

    # Single run with save
    python -m src.scripts.clustering.cluster \\
        --data-dir .../pretraining/<model>/ \\
        --embedding probs --preprocess mean_pca_l2 \\
        --method spherical_kmeans --k 64 --save

    # List available options
    python -m src.scripts.clustering.cluster --list
"""

import argparse
import json
import logging
import os
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

from src.scripts.clustering.transform import (
    EMBEDDING_FILES,
    PREPROCESS_REGISTRY,
    apply_preprocess,
    load_embedding,
)


def balance_by_class(
    emb: np.ndarray,
    meta: list,
    key: str,
    n: int | None,
    seed: int,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Stratified subsample: keep at most ``n`` rows per unique value of meta[key].

    If ``n`` is None or larger than a class's count, all rows for that class
    are kept. Returns ``(emb_sub, meta_sub, kept_indices)`` with the original
    row order preserved.
    """
    rng = np.random.default_rng(seed)
    by_class: dict = {}
    for i, m in enumerate(meta):
        if key not in m:
            raise KeyError(
                f"balance_by_class: metadata row {i} has no '{key}' field. "
                f"Keys present: {sorted(m.keys())}"
            )
        by_class.setdefault(m[key], []).append(i)

    counts = {cls: len(idxs) for cls, idxs in by_class.items()}
    min_cnt = min(counts.values())
    max_cnt = max(counts.values())
    target = n if n is not None else min_cnt
    logger.info(
        f"Balancing by '{key}': {len(counts)} classes, "
        f"min={min_cnt}, max={max_cnt}  ->  cap={target}/class"
    )

    kept: list[int] = []
    for cls in sorted(by_class.keys()):
        idxs = by_class[cls]
        if len(idxs) <= target:
            kept.extend(idxs)
        else:
            chosen = rng.choice(idxs, size=target, replace=False)
            kept.extend(chosen.tolist())

    kept.sort()
    kept_arr = np.asarray(kept, dtype=np.int64)
    post_counts: dict = {}
    meta_sub = [meta[i] for i in kept]
    for m in meta_sub:
        post_counts[m[key]] = post_counts.get(m[key], 0) + 1
    logger.info(
        f"  Balanced: {len(kept_arr)} rows "
        f"(min={min(post_counts.values())}, max={max(post_counts.values())})"
    )
    return emb[kept_arr], meta_sub, kept_arr


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

CLUSTER_REGISTRY = {}


def register_cluster(name: str, description: str):
    def decorator(fn):
        CLUSTER_REGISTRY[name] = {"fn": fn, "description": description}
        return fn

    return decorator


@register_cluster("kmeans", "MiniBatchKMeans")
def cluster_kmeans(emb: np.ndarray, k: int, **kwargs) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=k,
        n_init=10,
        max_iter=500,
        batch_size=4096,
        random_state=42,
    )
    labels = km.fit_predict(emb)
    logger.info(f"  KMeans: inertia={km.inertia_:.1f}")
    return labels


@register_cluster("spherical_kmeans", "Spherical KMeans (normalize centroids)")
def cluster_spherical_kmeans(emb: np.ndarray, k: int, **kwargs) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    emb_normed = normalize(emb, norm="l2")
    km = MiniBatchKMeans(
        n_clusters=k,
        n_init=10,
        max_iter=500,
        batch_size=4096,
        random_state=42,
    )
    km.fit(emb_normed)

    for iteration in range(20):
        old_centers = km.cluster_centers_.copy()
        km.cluster_centers_ = normalize(km.cluster_centers_, norm="l2")
        labels = km.predict(emb_normed)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                km.cluster_centers_[c] = normalize(
                    emb_normed[mask].mean(axis=0, keepdims=True), norm="l2"
                )[0]
        delta = np.abs(km.cluster_centers_ - old_centers).max()
        if delta < 1e-6:
            logger.info(f"  Spherical KMeans: converged at iteration {iteration + 1}")
            break

    labels = km.predict(emb_normed)
    logger.info(f"  Spherical KMeans: k={k}, {iteration + 1} refinement iterations")
    return labels


@register_cluster("hierarchical", "Agglomerative (precomputed distances)")
def cluster_hierarchical(emb: np.ndarray, k: int, **kwargs) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster

    Z = kwargs["linkage_matrix"]
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 1-based -> 0-based
    n_actual = len(set(labels))
    logger.info(f"  Hierarchical: cut at k={k}, got {n_actual} clusters")
    return labels


@register_cluster("gmm", "Gaussian Mixture Model")
def cluster_gmm(emb: np.ndarray, k: int, **kwargs) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        max_iter=200,
        n_init=3,
        random_state=42,
    )
    labels = gmm.fit_predict(emb)
    logger.info(f"  GMM: converged={gmm.converged_}, BIC={gmm.bic(emb):.1f}")
    return labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_clustering(emb: np.ndarray, labels: np.ndarray, meta: list) -> dict:
    """Compute clustering quality metrics."""
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_clusters = len(set(labels))
    metrics: dict[str, float | int] = {"n_clusters": n_clusters, "n_samples": len(labels)}

    if n_clusters < 2:
        logger.warning("  Only 1 cluster — skipping metrics")
        return metrics

    # Subsample for silhouette (expensive)
    n_sample = min(10_000, len(emb))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(emb), n_sample, replace=False)

    metrics["silhouette"] = float(silhouette_score(emb[idx], labels[idx], metric="euclidean"))
    metrics["silhouette_cosine"] = float(silhouette_score(emb[idx], labels[idx], metric="cosine"))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(emb, labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(emb, labels))

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    metrics["cluster_size_min"] = int(counts.min())
    metrics["cluster_size_max"] = int(counts.max())
    metrics["cluster_size_median"] = int(np.median(counts))
    metrics["cluster_size_std"] = float(counts.std())

    # Source entropy per cluster
    sources = np.array([m["source"] for m in meta])
    unique_sources = np.unique(sources)
    entropies = []
    for c in unique:
        mask = labels == c
        if mask.sum() == 0:
            continue
        c_sources = sources[mask]
        probs = np.array([np.sum(c_sources == s) for s in unique_sources], dtype=np.float64)
        probs = probs / probs.sum()
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))
    metrics["avg_source_entropy"] = float(np.mean(entropies))

    return metrics


def print_metrics(metrics: dict):
    logger.info("--- Metrics ---")
    logger.info(f"  n_clusters:        {metrics['n_clusters']}")
    logger.info(f"  n_samples:         {metrics['n_samples']}")
    if "silhouette" in metrics:
        logger.info(f"  silhouette:        {metrics['silhouette']:.4f}")
    if "silhouette_cosine" in metrics:
        logger.info(f"  silhouette_cosine: {metrics['silhouette_cosine']:.4f}")
    if "calinski_harabasz" in metrics:
        logger.info(f"  calinski_harabasz: {metrics['calinski_harabasz']:.1f}")
    if "davies_bouldin" in metrics:
        logger.info(f"  davies_bouldin:    {metrics['davies_bouldin']:.4f}")
    if "cluster_size_median" in metrics:
        logger.info(
            f"  cluster sizes:     min={metrics['cluster_size_min']}, "
            f"max={metrics['cluster_size_max']}, "
            f"median={metrics['cluster_size_median']}, "
            f"std={metrics['cluster_size_std']:.1f}"
        )
    if "avg_source_entropy" in metrics:
        logger.info(f"  avg_source_entropy: {metrics['avg_source_entropy']:.4f} bits")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(
    labels, metrics, meta, emb, transformed, output_dir, embedding, preprocess, method, k
):
    """Save assignments, metrics, and per-cluster summary."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "assignments.npy"), labels)

    run_info = {
        "embedding": embedding,
        "preprocess": preprocess,
        "method": method,
        "k": k,
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # Per-cluster summary
    sources = [m["source"] for m in meta]
    unique_sources = sorted(set(sources))
    summaries: list[dict[str, Any]] = []

    for c in range(k):
        mask = labels == c
        c_indices = np.where(mask)[0]
        c_size = int(mask.sum())
        if c_size == 0:
            summaries.append({"cluster": c, "size": 0})
            continue

        # Source distribution
        c_sources = [sources[i] for i in c_indices]
        source_counts = {s: c_sources.count(s) for s in unique_sources if c_sources.count(s) > 0}

        # Top activated experts
        cluster_emb_sum = emb[c_indices].sum(axis=0)
        top10_experts = np.argsort(cluster_emb_sum)[::-1][:10].tolist()

        # Representative samples (closest to centroid)
        centroid = transformed[c_indices].mean(axis=0)
        dists = np.linalg.norm(transformed[c_indices] - centroid, axis=1)
        closest = np.argsort(dists)[:5]
        rep_samples = []
        for idx in closest:
            global_idx = int(c_indices[idx])
            m = meta[global_idx]
            entry = {"idx": global_idx, "source": m["source"]}
            for key in (
                "doc_len",
                "preview",
                "token_position",
                "token_id",
                "doc_index",
                "category",
            ):
                if key in m:
                    entry[key] = m[key]
            rep_samples.append(entry)

        summaries.append(
            {
                "cluster": c,
                "size": c_size,
                "source_counts": source_counts,
                "top10_experts_global": top10_experts,
                "representative_samples": rep_samples,
            }
        )

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    logger.info(f"  Saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Hierarchical clustering helpers
# ---------------------------------------------------------------------------


def compute_distance_matrix(emb, metric="cosine"):
    """Compute condensed pairwise distance matrix."""
    import time

    from sklearn.preprocessing import normalize

    N = emb.shape[0]
    logger.info(f"Computing {metric} distances for {N} samples ...")
    t0 = time.time()

    if metric == "cosine":
        X_norm = normalize(emb, norm="l2").astype(np.float64)
        sim = X_norm @ X_norm.T
        np.clip(sim, -1.0, 1.0, out=sim)
        dist_sq = 1.0 - sim
        np.fill_diagonal(dist_sq, 0.0)
        from scipy.spatial.distance import squareform

        dist = squareform(dist_sq, checks=False)
    elif metric == "euclidean":
        X = emb.astype(np.float64)
        sq_norms = (X**2).sum(axis=1)
        gram = X @ X.T
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
        np.maximum(dist_sq, 0.0, out=dist_sq)
        dist_sq = np.sqrt(dist_sq)
        np.fill_diagonal(dist_sq, 0.0)
        from scipy.spatial.distance import squareform

        dist = squareform(dist_sq, checks=False)
    else:
        from scipy.spatial.distance import pdist

        dist = pdist(emb, metric=metric).astype(np.float64)

    logger.info(f"  Computed in {time.time() - t0:.1f}s")
    return dist


def compute_linkage(dist_condensed, method="average"):
    import time

    from scipy.cluster.hierarchy import linkage

    logger.info(f"  Computing linkage (method={method}) ...")
    t0 = time.time()
    Z = linkage(dist_condensed, method=method)
    logger.info(f"  Linkage in {time.time() - t0:.1f}s")
    return Z


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cluster router embeddings")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Directory with embeddings and info.json"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="probs",
        help=f"Embedding to cluster. "
        f"Available: {', '.join(sorted(EMBEDDING_FILES))} "
        f"(default: probs)",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="mean_pca_l2",
        help=f"Preprocessing. "
        f"Available: {', '.join(sorted(PREPROCESS_REGISTRY))} "
        f"(default: mean_pca_l2)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="spherical_kmeans",
        help=f"Clustering method. "
        f"Available: {', '.join(sorted(CLUSTER_REGISTRY))} "
        f"(default: spherical_kmeans)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[64],
        help="Number of clusters. Multiple values for sweep. (default: 64)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save results (assignments, metrics, summary)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output dir (default: auto-named subdirectory)",
    )

    # Hierarchical-specific
    parser.add_argument(
        "--dist-metric",
        type=str,
        default="cosine",
        help="Distance metric for hierarchical (default: cosine)",
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default="average",
        help="Linkage method for hierarchical (default: average)",
    )

    # Class balancing (stratified subsample before preprocessing)
    parser.add_argument(
        "--balance-by",
        type=str,
        default=None,
        help=(
            "Metadata key to balance by (e.g., 'source'). If unset, no "
            "balancing. Applied after load, before preprocess, so PCA is fit "
            "on the balanced set."
        ),
    )
    parser.add_argument(
        "--balance-n",
        type=int,
        default=None,
        help=(
            "Per-class cap when --balance-by is set. Default = min count "
            "across classes. Classes with fewer rows are kept in full."
        ),
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Seed for stratified subsample (default: 42).",
    )

    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nEmbeddings:")
        for name in sorted(EMBEDDING_FILES):
            print(f"  {name}")
        print("\nPreprocessing:")
        for name, entry in sorted(PREPROCESS_REGISTRY.items()):
            print(f"  {name:20s}  {entry['description']}")
        print("\nClustering methods:")
        for name, entry in sorted(CLUSTER_REGISTRY.items()):
            print(f"  {name:20s}  {entry['description']}")
        return

    if not args.data_dir:
        parser.error("--data-dir is required")

    # Load info (always needed)
    info_path = os.path.join(args.data_dir, "info.json")
    with open(info_path) as f:
        info = json.load(f)

    # Cache key: include balancing so balanced/unbalanced runs don't collide
    if args.balance_by is not None:
        bal_tag = f"_bal{args.balance_by}N{args.balance_n}seed{args.balance_seed}"
    else:
        bal_tag = ""
    cache_path = os.path.join(
        args.data_dir,
        f"preprocessed_{args.embedding}_{args.preprocess}{bal_tag}.npy",
    )
    # When balancing, also cache the subsampled meta so we can reload without
    # redoing the stratified draw.
    meta_cache_path = cache_path.replace(".npy", ".meta.json") if bal_tag else None

    emb = None
    transformed: np.ndarray
    meta: list

    if os.path.exists(cache_path) and (meta_cache_path is None or os.path.exists(meta_cache_path)):
        logger.info(f"Loading cached preprocessed: {cache_path}")
        transformed = np.load(cache_path)
        logger.info(f"  shape={transformed.shape}")
        if meta_cache_path:
            with open(meta_cache_path) as f:
                meta = json.load(f)
            logger.info(f"  loaded balanced meta ({len(meta)} rows)")
        else:
            emb_loaded, meta, _ = load_embedding(args.data_dir, args.embedding)
            if args.save:
                emb = emb_loaded
            else:
                del emb_loaded
        if args.save and emb is None:
            emb, full_meta, _ = load_embedding(args.data_dir, args.embedding)
            if args.balance_by is not None:
                # Deterministic reapply so emb rows align with cached transformed.
                emb, _, _ = balance_by_class(
                    emb,
                    full_meta,
                    args.balance_by,
                    args.balance_n,
                    args.balance_seed,
                )
                assert emb.shape[0] == transformed.shape[0], (
                    f"Balance reapply mismatch: emb {emb.shape[0]} vs cached "
                    f"transformed {transformed.shape[0]}"
                )
    else:
        emb, meta, _ = load_embedding(args.data_dir, args.embedding)
        if args.balance_by is not None:
            emb, meta, _ = balance_by_class(
                emb, meta, args.balance_by, args.balance_n, args.balance_seed
            )
        transformed = apply_preprocess(emb, args.preprocess, info).astype(np.float32)
        logger.info(f"Saving preprocess cache: {cache_path}  shape={transformed.shape}")
        np.save(cache_path, transformed)
        if meta_cache_path:
            with open(meta_cache_path, "w") as f:
                json.dump(meta, f)
            logger.info(f"  saved balanced meta: {meta_cache_path}")
        if not args.save:
            del emb
            emb = None
    logger.info(f"  Preprocessed shape: {transformed.shape}")

    # Hierarchical: precompute distance + linkage once
    cluster_kwargs = {}
    if args.method == "hierarchical":
        dist = compute_distance_matrix(transformed, args.dist_metric)
        linkage_matrix = compute_linkage(dist, args.linkage_method)
        cluster_kwargs["linkage_matrix"] = linkage_matrix

    # Cluster + evaluate for each k
    all_results = []
    for k in args.k:
        logger.info(f"\n--- k={k} ---")
        if args.method not in CLUSTER_REGISTRY:
            raise ValueError(
                f"Unknown method '{args.method}'. "
                f"Available: {', '.join(sorted(CLUSTER_REGISTRY))}"
            )

        labels = CLUSTER_REGISTRY[args.method]["fn"](transformed, k, **cluster_kwargs)
        metrics = evaluate_clustering(transformed, labels, meta)
        print_metrics(metrics)
        all_results.append({"k": k, **metrics})

        if args.save:
            output_dir = args.output_dir or os.path.join(
                args.data_dir, f"{args.embedding}_{args.preprocess}_{args.method}_k{k}"
            )
            save_results(
                labels,
                metrics,
                meta,
                emb,
                transformed,
                output_dir,
                args.embedding,
                args.preprocess,
                args.method,
                k,
            )

    # Sweep summary table
    if len(args.k) > 1:
        logger.info("\n=== SWEEP SUMMARY ===")
        header = (
            f"{'k':>5}  {'sil_euc':>8}  {'sil_cos':>8}  "
            f"{'CH':>10}  {'DB':>8}  {'sz_med':>7}  {'src_ent':>8}"
        )
        logger.info(header)
        logger.info("-" * len(header))
        for r in all_results:
            logger.info(
                f"{r['k']:>5}  "
                f"{r.get('silhouette', float('nan')):>8.4f}  "
                f"{r.get('silhouette_cosine', float('nan')):>8.4f}  "
                f"{r.get('calinski_harabasz', float('nan')):>10.1f}  "
                f"{r.get('davies_bouldin', float('nan')):>8.4f}  "
                f"{r.get('cluster_size_median', 0):>7}  "
                f"{r.get('avg_source_entropy', 0):>8.4f}"
            )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
