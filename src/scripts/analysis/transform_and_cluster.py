"""
Load a router embedding, apply a transformation, and cluster.

Usage:
    python -m src.scripts.analysis.transform_and_cluster \
        --data-dir claude_outputs/analysis/router_clustering_pretraining \
        --embedding logits \
        --transform mean_pca_l2 \
        --cluster kmeans --k 64

    # List available embeddings, transforms, and clustering algorithms
    python -m src.scripts.analysis.transform_and_cluster --list
"""

import argparse
import json
import logging
import os

# Limit BLAS threads before numpy/sklearn imports
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

# Maps embedding name -> filename
EMBEDDING_FILES = {
    "logits": "embeddings_logits.npy",
    "probs": "embeddings_probs.npy",
    "logits_sparse": "embeddings_logits_sparse.npy",
    "probs_sparse": "embeddings_probs_sparse.npy",
    "topk_freq": "embeddings_topk_freq.npy",
    # Token-level embeddings (from --granularity token)
    "token_logits": "embeddings_token_logits.npy",
    "token_probs": "embeddings_token_probs.npy",
    "token_topk_binary": "embeddings_token_topk_binary.npy",
}


def load_embedding(data_dir: str, embedding_name: str) -> tuple:
    """
    Load an embedding array and its associated metadata.

    Returns:
        emb: (N, D) float32 array
        meta: list of dicts (one per doc)
        info: dict with model/extraction metadata
    """
    if embedding_name not in EMBEDDING_FILES:
        raise ValueError(
            f"Unknown embedding '{embedding_name}'. "
            f"Available: {', '.join(sorted(EMBEDDING_FILES))}"
        )

    emb_path = os.path.join(data_dir, EMBEDDING_FILES[embedding_name])
    info_path = os.path.join(data_dir, "info.json")
    # Auto-detect metadata file: token-level uses metadata_tokens.jsonl.gz
    is_token = embedding_name.startswith("token_")
    meta_path = os.path.join(
        data_dir, "metadata_tokens.jsonl.gz" if is_token else "metadata.jsonl.gz"
    )

    logger.info(f"Loading embedding '{embedding_name}' from {emb_path}")
    emb = np.load(emb_path).astype(np.float32)

    with open(info_path) as f:
        info = json.load(f)

    import gzip

    meta = []
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    assert emb.shape[0] == len(
        meta
    ), f"Embedding rows ({emb.shape[0]}) != metadata rows ({len(meta)})"

    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    assert (
        emb.shape[1] == num_layers * num_experts
    ), f"Embedding dim ({emb.shape[1]}) != {num_layers} * {num_experts}"

    logger.info(f"  shape={emb.shape}, dtype=float32, " f"range=[{emb.min():.4f}, {emb.max():.4f}]")
    logger.info(f"  {len(meta)} docs, {num_layers} layers, {num_experts} experts")

    return emb, meta, info


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

# Each transform is a function: (emb, info) -> transformed_emb
# `info` dict has num_layers, num_standard_experts, etc.
TRANSFORM_REGISTRY = {}


def register_transform(name: str, description: str):
    """Decorator to register a transform function."""

    def decorator(fn):
        TRANSFORM_REGISTRY[name] = {"fn": fn, "description": description}
        return fn

    return decorator


@register_transform("identity", "No transform — raw embedding values as-is")
def transform_identity(emb: np.ndarray, info: dict) -> np.ndarray:
    return emb


@register_transform("l2", "L2 normalize each document vector")
def transform_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.preprocessing import normalize

    return normalize(emb, norm="l2")


@register_transform("mean_pca", "Mean-center then PCA (95% variance)")
def transform_mean_pca(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA

    # Mean-center
    centered = emb - emb.mean(axis=0, keepdims=True)

    # PCA
    n_components = min(centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(centered)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")

    pca_k = PCA(n_components=k, random_state=42)
    return pca_k.fit_transform(centered)


@register_transform("mean_pca_l2", "Mean-center then PCA (95% variance) then L2 normalize")
def transform_mean_pca_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    # Mean-center
    centered = emb - emb.mean(axis=0, keepdims=True)

    n_components = min(centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(centered)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")

    pca_k = PCA(n_components=k, random_state=42)
    reduced = pca_k.fit_transform(centered)
    return normalize(reduced, norm="l2")


@register_transform("mean_l2_pca", "Mean-center, L2 normalize, then PCA (95% variance)")
def transform_mean_l2_pca(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    # Mean-center
    centered = emb - emb.mean(axis=0, keepdims=True)

    # L2 normalize
    normalized = normalize(centered, norm="l2")

    # PCA
    n_components = min(normalized.shape[0], normalized.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(normalized)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")

    pca_k = PCA(n_components=k, random_state=42)
    return pca_k.fit_transform(normalized)


@register_transform("l2_mean_pca", "L2 normalize, mean-center, then PCA (95% variance)")
def transform_l2_mean_pca(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    # L2 normalize
    normalized = normalize(emb, norm="l2")

    # Mean-center
    centered = normalized - normalized.mean(axis=0, keepdims=True)

    # PCA
    n_components = min(centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(centered)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")

    pca_k = PCA(n_components=k, random_state=42)
    return pca_k.fit_transform(centered)


@register_transform("l2_mean_pca_l2", "L2 normalize, mean-center, PCA (95% variance), L2 normalize")
def transform_l2_mean_pca_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    # L2 normalize
    normalized = normalize(emb, norm="l2")

    # Mean-center
    centered = normalized - normalized.mean(axis=0, keepdims=True)

    # PCA
    n_components = min(centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(centered)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")

    pca_k = PCA(n_components=k, random_state=42)
    reduced = pca_k.fit_transform(centered)

    # L2 normalize again
    return normalize(reduced, norm="l2")


@register_transform("tsvd", "TruncatedSVD (95% variance)")
def transform_tsvd(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    n_components = min(emb.shape[0], emb.shape[1])
    tsvd = TruncatedSVD(n_components=n_components, random_state=42)
    tsvd.fit(emb)

    cumvar = np.cumsum(tsvd.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  TruncatedSVD: {k} components explain {cumvar[k-1]:.1%} variance")

    tsvd_k = TruncatedSVD(n_components=k, random_state=42)
    return tsvd_k.fit_transform(emb)


@register_transform("l2_tsvd", "L2 normalize then TruncatedSVD (95% variance)")
def transform_l2_tsvd(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    # L2 normalize
    normalized = normalize(emb, norm="l2")

    n_components = min(normalized.shape[0], normalized.shape[1])
    tsvd = TruncatedSVD(n_components=n_components, random_state=42)
    tsvd.fit(normalized)

    cumvar = np.cumsum(tsvd.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  TruncatedSVD: {k} components explain {cumvar[k-1]:.1%} variance")

    tsvd_k = TruncatedSVD(n_components=k, random_state=42)
    return tsvd_k.fit_transform(normalized)


@register_transform("tsvd_l2", "TruncatedSVD (95% variance) then L2 normalize")
def transform_tsvd_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    n_components = min(emb.shape[0], emb.shape[1])
    tsvd = TruncatedSVD(n_components=n_components, random_state=42)
    tsvd.fit(emb)

    cumvar = np.cumsum(tsvd.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  TruncatedSVD: {k} components explain {cumvar[k-1]:.1%} variance")

    tsvd_k = TruncatedSVD(n_components=k, random_state=42)
    reduced = tsvd_k.fit_transform(emb)
    return normalize(reduced, norm="l2")


def apply_transform(emb: np.ndarray, transform_name: str, info: dict) -> np.ndarray:
    """Apply a named transform to an embedding array."""
    if transform_name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{transform_name}'. "
            f"Available: {', '.join(sorted(TRANSFORM_REGISTRY))}"
        )
    logger.info(f"Applying transform: {transform_name}")
    return TRANSFORM_REGISTRY[transform_name]["fn"](emb, info)


# ---------------------------------------------------------------------------
# Clustering registry
# ---------------------------------------------------------------------------

CLUSTER_REGISTRY = {}


def register_cluster(name: str, description: str):
    """Decorator to register a clustering algorithm."""

    def decorator(fn):
        CLUSTER_REGISTRY[name] = {"fn": fn, "description": description}
        return fn

    return decorator


@register_cluster("kmeans", "MiniBatchKMeans clustering")
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


@register_cluster("spherical_kmeans", "Spherical KMeans (L2-normalize centroids each iteration)")
def cluster_spherical_kmeans(emb: np.ndarray, k: int, **kwargs) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    # Normalize input to unit sphere
    emb_normed = normalize(emb, norm="l2")

    km = MiniBatchKMeans(
        n_clusters=k,
        n_init=10,
        max_iter=500,
        batch_size=4096,
        random_state=42,
    )

    # Iterative: fit, normalize centroids, reassign
    km.fit(emb_normed)
    for iteration in range(20):
        old_centers = km.cluster_centers_.copy()
        km.cluster_centers_ = normalize(km.cluster_centers_, norm="l2")
        labels = km.predict(emb_normed)
        # Refit centroids from assignments
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                km.cluster_centers_[c] = normalize(
                    emb_normed[mask].mean(axis=0, keepdims=True), norm="l2"
                )[0]
        # Check convergence
        delta = np.abs(km.cluster_centers_ - old_centers).max()
        if delta < 1e-6:
            logger.info(f"  Spherical KMeans: converged at iteration {iteration + 1}")
            break

    labels = km.predict(emb_normed)
    logger.info(f"  Spherical KMeans: k={k}, {iteration + 1} refinement iterations")
    return labels


# ---------------------------------------------------------------------------
# Hierarchical clustering helpers
# ---------------------------------------------------------------------------


def _normalize_to_distribution(arr: np.ndarray) -> np.ndarray:
    """Normalize each row to sum to 1 (for JSD). Clips negatives to 0."""
    arr = np.clip(arr, 0, None)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return arr / row_sums


def _pairwise_to_condensed(square: np.ndarray) -> np.ndarray:
    """Convert a square distance matrix to scipy condensed form (upper triangle)."""
    from scipy.spatial.distance import squareform

    return squareform(square, checks=False)


def _fast_pairwise(data: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute pairwise distances using fast BLAS-based implementations.

    Returns condensed distance matrix (N*(N-1)/2,) as float64.

    - cosine: 1 - X_norm @ X_norm.T (BLAS matmul)
    - euclidean: sqrt(||x||^2 + ||y||^2 - 2*x.y) via BLAS matmul
    - jensenshannon: scipy pdist (C-optimized, single-threaded — no good
      parallel option without OpenBLAS thread issues)
    """
    # N = data.shape[0]

    if metric == "cosine":
        from sklearn.preprocessing import normalize

        X_norm = normalize(data, norm="l2").astype(np.float64)
        sim = X_norm @ X_norm.T
        np.clip(sim, -1.0, 1.0, out=sim)
        dist_square = 1.0 - sim
        np.fill_diagonal(dist_square, 0.0)
        return _pairwise_to_condensed(dist_square)

    elif metric == "euclidean":
        # BLAS matmul: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X = data.astype(np.float64)
        sq_norms = (X**2).sum(axis=1)
        gram = X @ X.T
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
        np.maximum(dist_sq, 0.0, out=dist_sq)  # numerical safety
        dist_square = np.sqrt(dist_sq)
        np.fill_diagonal(dist_square, 0.0)
        return _pairwise_to_condensed(dist_square)

    elif metric == "jensenshannon":
        from scipy.spatial.distance import pdist

        return pdist(data, metric="jensenshannon").astype(np.float64)

    else:
        from scipy.spatial.distance import pdist

        return pdist(data, metric=metric).astype(np.float64)


def compute_distance_matrix(
    emb: np.ndarray,
    dist_metric: str,
    dist_mode: str,
    num_layers: int,
    num_experts: int,
) -> np.ndarray:
    """
    Compute condensed pairwise distance matrix.

    Uses optimized implementations:
    - cosine: BLAS matrix multiply (1 - X_norm @ X_norm.T)
    - euclidean: sklearn parallel pairwise_distances
    - jensenshannon: sklearn parallel with scipy JSD callable

    Args:
        emb: (N, num_layers * num_experts) embedding array
        dist_metric: 'cosine', 'euclidean', or 'jensenshannon'
        dist_mode: 'flat' (whole vector) or 'per_layer' (average per-layer distances)
        num_layers, num_experts: MoE dimensions

    Returns:
        Condensed distance matrix of shape (N*(N-1)/2,) as float64.
    """
    import time

    N = emb.shape[0]
    n_pairs = N * (N - 1) // 2
    logger.info(
        f"Computing {dist_metric} distance ({dist_mode} mode) "
        f"for {N} docs ({n_pairs:,} pairs) ..."
    )

    if dist_mode == "flat":
        data = emb
        if dist_metric == "jensenshannon":
            data = _normalize_to_distribution(data)
        t0 = time.time()
        dist = _fast_pairwise(data, dist_metric)
        logger.info(f"  Flat distance computed in {time.time() - t0:.1f}s")

    elif dist_mode == "per_layer":
        reshaped = emb.reshape(N, num_layers, num_experts)
        dist_sum = np.zeros(n_pairs, dtype=np.float64)

        t0 = time.time()
        for layer_idx in range(num_layers):
            layer_data = reshaped[:, layer_idx, :].copy()
            if dist_metric == "jensenshannon":
                layer_data = _normalize_to_distribution(layer_data)
            layer_dist = _fast_pairwise(layer_data, dist_metric)
            dist_sum += layer_dist

            if (layer_idx + 1) % 4 == 0 or layer_idx == num_layers - 1:
                elapsed = time.time() - t0
                logger.info(f"  Layer {layer_idx + 1}/{num_layers} done ({elapsed:.1f}s)")

        dist = dist_sum / num_layers
        logger.info(f"  Per-layer distance computed in {time.time() - t0:.1f}s")
    else:
        raise ValueError(f"Unknown dist_mode '{dist_mode}'. Use 'flat' or 'per_layer'.")

    logger.info(
        f"  Distance matrix: {dist.shape[0]:,} pairs, "
        f"range=[{dist.min():.4f}, {dist.max():.4f}], "
        f"mean={dist.mean():.4f}, memory={dist.nbytes / 1e9:.1f}GB"
    )
    return dist


def compute_linkage(dist_condensed: np.ndarray, linkage_method: str) -> np.ndarray:
    """Compute hierarchical clustering linkage matrix from condensed distances."""
    import time

    from scipy.cluster.hierarchy import linkage

    logger.info(f"  Computing linkage (method={linkage_method}) ...")
    t0 = time.time()
    Z = linkage(dist_condensed, method=linkage_method)
    logger.info(f"  Linkage computed in {time.time() - t0:.1f}s")
    return Z


@register_cluster(
    "hierarchical", "Hierarchical agglomerative clustering with precomputed distances"
)
def cluster_hierarchical(emb: np.ndarray, k: int, **kwargs) -> np.ndarray:
    """
    Hierarchical clustering. Requires kwargs:
        linkage_matrix: precomputed linkage matrix (from compute_linkage)
    """
    from scipy.cluster.hierarchy import fcluster

    Z = kwargs["linkage_matrix"]
    labels = fcluster(Z, t=k, criterion="maxclust")
    labels = labels - 1  # fcluster returns 1-based labels
    n_actual = len(set(labels))
    logger.info(f"  Hierarchical: cut at k={k}, got {n_actual} actual clusters")
    return labels


@register_cluster("gmm", "Gaussian Mixture Model clustering")
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
    logger.info(
        f"  GMM: converged={gmm.converged_}, " f"BIC={gmm.bic(emb):.1f}, AIC={gmm.aic(emb):.1f}"
    )
    return labels


def run_clustering(emb: np.ndarray, cluster_name: str, k: int, **kwargs) -> np.ndarray:
    """Run a named clustering algorithm. Returns (N,) int array of labels."""
    if cluster_name not in CLUSTER_REGISTRY:
        raise ValueError(
            f"Unknown clustering '{cluster_name}'. "
            f"Available: {', '.join(sorted(CLUSTER_REGISTRY))}"
        )
    logger.info(f"Clustering: {cluster_name} with k={k}")
    return CLUSTER_REGISTRY[cluster_name]["fn"](emb, k, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def evaluate_clustering(
    emb: np.ndarray,
    labels: np.ndarray,
    meta: list,
    precomputed_dist: np.ndarray = None,
) -> dict:
    """
    Compute clustering quality metrics.

    If precomputed_dist is provided (condensed distance matrix), silhouette is
    computed using that distance. Otherwise, euclidean and cosine silhouette are
    computed from the embedding vectors.

    Returns a dict of metric_name -> value.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_clusters = len(set(labels))
    metrics = {"n_clusters": n_clusters, "n_docs": len(labels)}

    if n_clusters < 2:
        logger.warning("  Only 1 cluster — skipping all metrics")
        return metrics

    n_sample = min(10_000, len(emb))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(emb), n_sample, replace=False)

    if precomputed_dist is not None:
        # Use precomputed distance matrix for silhouette
        from scipy.spatial.distance import squareform

        dist_square = squareform(precomputed_dist)
        idx_sorted = np.sort(idx)
        sub_dist = dist_square[np.ix_(idx_sorted, idx_sorted)]
        sub_labels = labels[idx_sorted]
        metrics["silhouette"] = float(
            silhouette_score(sub_dist, sub_labels, metric="precomputed", sample_size=None)
        )
        # No separate cosine silhouette — the precomputed distance already
        # encodes the chosen metric
    else:
        # Silhouette score: [-1, 1], higher = better separated clusters
        metrics["silhouette"] = float(
            silhouette_score(emb[idx], labels[idx], metric="euclidean", sample_size=None)
        )
        metrics["silhouette_cosine"] = float(
            silhouette_score(emb[idx], labels[idx], metric="cosine", sample_size=None)
        )

    # Calinski-Harabasz index: higher = denser, well-separated clusters
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(emb, labels))

    # Davies-Bouldin index: lower = better (measures avg cluster similarity)
    metrics["davies_bouldin"] = float(davies_bouldin_score(emb, labels))

    # Cluster size stats
    unique, counts = np.unique(labels, return_counts=True)
    metrics["cluster_size_min"] = int(counts.min())
    metrics["cluster_size_max"] = int(counts.max())
    metrics["cluster_size_median"] = int(np.median(counts))
    metrics["cluster_size_std"] = float(counts.std())

    # Source entropy: how mixed are clusters w.r.t. data sources?
    # Higher = more mixing (clusters are not just source buckets)
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
    """Pretty-print evaluation metrics."""
    logger.info("--- Evaluation Metrics ---")
    logger.info(f"  n_clusters:          {metrics['n_clusters']}")
    logger.info(f"  n_docs:              {metrics['n_docs']}")
    if "silhouette" in metrics:
        logger.info(f"  silhouette:          {metrics['silhouette']:.4f}  (higher=better, [-1,1])")
    if "silhouette_cosine" in metrics:
        logger.info(
            f"  silhouette_cosine:   {metrics['silhouette_cosine']:.4f}  (higher=better, [-1,1])"
        )
        logger.info(f"  calinski_harabasz:   {metrics['calinski_harabasz']:.1f}  (higher=better)")
        logger.info(f"  davies_bouldin:      {metrics['davies_bouldin']:.4f}  (lower=better)")
        logger.info(
            f"  cluster sizes:       min={metrics['cluster_size_min']}, "
            f"max={metrics['cluster_size_max']}, "
            f"median={metrics['cluster_size_median']}, "
            f"std={metrics['cluster_size_std']:.1f}"
        )
        logger.info(f"  avg_source_entropy:  {metrics['avg_source_entropy']:.4f} bits")


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def save_results(
    labels: np.ndarray,
    metrics: dict,
    meta: list,
    emb: np.ndarray,
    transformed: np.ndarray,
    args,
    output_dir: str,
    n_rep_docs: int = 5,
):
    """Save clustering assignments, metrics, and a summary report.

    Parameters
    ----------
    emb : raw embedding array (N, num_layers * num_experts) — used for top-expert computation
    transformed : transformed embedding array — used to find representative docs (closest to centroid)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Assignments
    np.save(os.path.join(output_dir, "assignments.npy"), labels)

    # Metrics + run config
    run_info = {
        "embedding": args.embedding,
        "transform": args.transform,
        "cluster": args.cluster,
        "k": args.k,
        "layers": args.layers,
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # Per-cluster summary (with top experts and representative docs)
    sources = [m["source"] for m in meta]
    unique_sources = sorted(set(sources))
    summaries = []
    for c in range(args.k):
        mask = labels == c
        c_indices = np.where(mask)[0]
        c_size = int(mask.sum())
        if c_size == 0:
            summaries.append(
                {
                    "cluster": c,
                    "size": 0,
                    "source_counts": {},
                    "top10_experts_global": [],
                    "representative_docs": [],
                }
            )
            continue

        # Source counts
        c_sources = [sources[i] for i in c_indices]
        source_counts = {s: c_sources.count(s) for s in unique_sources if c_sources.count(s) > 0}

        # Top 10 experts by summed activation across cluster docs
        cluster_emb_sum = emb[c_indices].sum(axis=0)  # (num_layers * num_experts,)
        top10_experts = np.argsort(cluster_emb_sum)[::-1][:10].tolist()

        # Representative docs: closest to centroid in transformed space
        centroid = transformed[c_indices].mean(axis=0)
        dists = np.linalg.norm(transformed[c_indices] - centroid, axis=1)
        closest = np.argsort(dists)[:n_rep_docs]
        rep_docs = []
        for idx in closest:
            global_idx = int(c_indices[idx])
            m = meta[global_idx]
            entry = {
                "idx": global_idx,
                "doc_index": global_idx,
                "source": m["source"],
            }
            # Document-level metadata has doc_len/preview; token-level has token_position/token_id
            if "doc_len" in m:
                entry["doc_len"] = m["doc_len"]
            if "preview" in m:
                entry["preview"] = m["preview"][:3000]
            if "token_position" in m:
                entry["token_position"] = m["token_position"]
            if "token_id" in m:
                entry["token_id"] = m["token_id"]
            rep_docs.append(entry)

        summaries.append(
            {
                "cluster": c,
                "size": c_size,
                "source_counts": source_counts,
                "top10_experts_global": top10_experts,
                "representative_docs": rep_docs,
            }
        )

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    logger.info(f"  Saved results to {output_dir}/")
    logger.info("    assignments.npy, run_info.json, summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Load router embeddings, apply transformations, and cluster."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing embeddings, metadata, and info.json "
        "(e.g. claude_outputs/analysis/router_clustering_pretraining/<model_name>)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="logits",
        help=f"Embedding type to load. "
        f"Available: {', '.join(sorted(EMBEDDING_FILES))} "
        f"(default: logits)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="identity",
        help=f"Transform to apply. "
        f"Available: {', '.join(sorted(TRANSFORM_REGISTRY))} "
        f"(default: identity)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="kmeans",
        help=f"Clustering algorithm. "
        f"Available: {', '.join(sorted(CLUSTER_REGISTRY))} "
        f"(default: kmeans)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[64],
        help="Number of clusters. Pass multiple values for sweep. " "(default: 64)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save clustering results (assignments, metrics, summary). "
        "Without this flag, only prints evaluation metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory when --save is set. "
        "Defaults to <data-dir>/<embedding>_<transform>_<cluster>_k<k>/",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layer range to use (e.g. '3-16' for layers 3..15, '11-16' for layers 11..15). "
        "Uses 1-indexed inclusive start, exclusive end. "
        "If not specified, uses all layers.",
    )
    parser.add_argument(
        "--dist-metric",
        type=str,
        default="cosine",
        help="Distance metric for hierarchical clustering: "
        "cosine, euclidean, jensenshannon (default: cosine)",
    )
    parser.add_argument(
        "--dist-mode",
        type=str,
        default="flat",
        help="Distance computation mode for hierarchical clustering: "
        "flat (whole vector) or per_layer (avg per-layer distances) "
        "(default: flat)",
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default="average",
        help="Linkage method for hierarchical clustering: " "average, complete (default: average)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available embeddings, transforms, and clusterers, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable embeddings:")
        for name, filename in sorted(EMBEDDING_FILES.items()):
            print(f"  {name:20s} -> {filename}")
        print("\nAvailable transforms:")
        for name, entry in sorted(TRANSFORM_REGISTRY.items()):
            print(f"  {name:20s} — {entry['description']}")
        print("\nAvailable clustering algorithms:")
        for name, entry in sorted(CLUSTER_REGISTRY.items()):
            print(f"  {name:20s} — {entry['description']}")
        return

    # Load
    emb, meta, info = load_embedding(args.data_dir, args.embedding)

    # Slice layers if requested
    if args.layers:
        parts = args.layers.split("-")
        layer_start = int(parts[0])
        layer_end = int(parts[1])
        num_experts = info["num_standard_experts"]
        num_layers = info["num_layers"]
        if layer_start < 0 or layer_end > num_layers or layer_start >= layer_end:
            raise ValueError(
                f"Invalid --layers '{args.layers}': must be <start>-<end> with "
                f"0 <= start < end <= {num_layers}"
            )
        col_start = layer_start * num_experts
        col_end = layer_end * num_experts
        emb = emb[:, col_start:col_end]
        n_selected = layer_end - layer_start
        logger.info(
            f"  Sliced layers {layer_start}-{layer_end} "
            f"({n_selected} layers): shape={emb.shape}"
        )
        # Update info so transforms see correct dimensions
        info = {**info, "num_layers": n_selected}

    # Transform
    transformed = apply_transform(emb, args.transform, info)
    logger.info(f"  Output shape: {transformed.shape}")

    # For hierarchical clustering: precompute distance matrix + linkage once
    dist_condensed = None
    linkage_matrix = None
    cluster_kwargs = {}
    if args.cluster == "hierarchical":
        # For per_layer mode, use the original (untransformed) embedding since
        # transforms like PCA destroy the layer structure.
        # For flat mode, use the transformed embedding.
        if args.dist_mode == "per_layer":
            if args.transform != "identity":
                logger.warning(
                    f"  per_layer distance mode ignores transform '{args.transform}' — "
                    f"using raw embedding to preserve layer structure"
                )
            dist_emb = emb
        else:
            dist_emb = transformed
        dist_condensed = compute_distance_matrix(
            dist_emb,
            args.dist_metric,
            args.dist_mode,
            info["num_layers"],
            info["num_standard_experts"],
        )
        linkage_matrix = compute_linkage(dist_condensed, args.linkage_method)
        cluster_kwargs["linkage_matrix"] = linkage_matrix

    # Cluster + evaluate for each k
    all_results = []
    for k in args.k:
        logger.info(f"\n--- k={k} ---")
        args_k = argparse.Namespace(**vars(args), **{"k_single": k})  # for save_results
        args_k.k = k

        labels = run_clustering(transformed, args.cluster, k, **cluster_kwargs)
        metrics = evaluate_clustering(
            transformed,
            labels,
            meta,
            precomputed_dist=dist_condensed,
        )
        print_metrics(metrics)
        all_results.append({"k": k, **metrics})

        if args.save:
            layers_suffix = f"_L{args.layers}" if args.layers else ""
            output_dir = args.output_dir or os.path.join(
                args.data_dir,
                f"{args.embedding}_{args.transform}_{args.cluster}_k{k}{layers_suffix}",
            )
            save_results(labels, metrics, meta, emb, transformed, args_k, output_dir)

    # Print summary table if multiple k values
    if len(args.k) > 1:
        logger.info("\n=== SWEEP SUMMARY ===")
        header = (
            f"{'k':>5}  {'silhouette':>11}  {'calinski_h':>11}  {'davies_b':>10}  "
            f"{'sz_med':>7}  {'sz_std':>8}  {'src_ent':>8}"
        )
        logger.info(header)
        logger.info("-" * len(header))
        for r in all_results:
            logger.info(
                f"{r['k']:>5}  "
                f"{r.get('silhouette', float('nan')):>11.4f}  "
                f"{r.get('calinski_harabasz', float('nan')):>11.1f}  "
                f"{r.get('davies_bouldin', float('nan')):>10.4f}  "
                f"{r.get('cluster_size_median', 0):>7}  "
                f"{r.get('cluster_size_std', 0):>8.1f}  "
                f"{r.get('avg_source_entropy', 0):>8.4f}"
            )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
