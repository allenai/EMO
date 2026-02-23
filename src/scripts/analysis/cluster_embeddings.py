"""
Cluster document router embeddings to discover implicit domains learned by the MoE model.

Steps:
  1. Load embeddings (optA/B/C/D .npy file)
  2. Apply a transform pipeline (see --transform)
  3. K-means sweep over k values → elbow + silhouette plots
  4. For chosen k: save assignments + per-cluster report

Transform options (--transform):
  raw                No transform — raw embedding values as-is
  l2                 L2 normalize only
  pca_l2             PCA (95% var) → L2 normalize  [default]
  log_l2             log1p → L2 normalize (compresses dynamic range)
  standardize_pca_l2 Z-score per feature → PCA (95% var) → L2 normalize
  renorm_l2          Per-layer renormalize to sum=1 → L2 normalize
  tsvd_l2            TruncatedSVD (95% var) → L2 normalize (no mean-centering)
  renorm_tsvd_l2     Per-layer renormalize → TruncatedSVD → L2 normalize

Usage:
    # Step 1: sweep to pick k (produces plots)
    python -m src.scripts.analysis.cluster_embeddings \
        --output-dir claude_outputs/analysis/router_clustering \
        --mode sweep --transform pca_l2

    # Step 2: run final clustering with chosen k
    python -m src.scripts.analysis.cluster_embeddings \
        --output-dir claude_outputs/analysis/router_clustering \
        --mode cluster --k 128 --transform pca_l2
"""

import argparse
import gzip
import json
import logging
import os

# Limit BLAS threads before any numpy/sklearn import to prevent OpenBLAS segfault
# on machines with many cores (>128 thread limit in precompiled OpenBLAS).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(output_dir: str, emb_file: str = None, data_dir: str = None):
    if data_dir is None:
        data_dir = output_dir
    if emb_file is None:
        emb_file = os.path.join(data_dir, "embeddings_optA_avgprob.npy")
    meta_path = os.path.join(data_dir, "metadata.jsonl.gz")
    info_path = os.path.join(data_dir, "info.json")

    logger.info(f"Loading embeddings from {emb_file} ...")
    emb = np.load(emb_file).astype(np.float32)  # (N, D)

    logger.info(f"Loading metadata ...")
    meta = []
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    with open(info_path) as f:
        info = json.load(f)

    logger.info(f"Loaded {emb.shape[0]} docs, embedding dim {emb.shape[1]}")
    return emb, meta, info


# ---------------------------------------------------------------------------
# PCA + normalize
# ---------------------------------------------------------------------------

def reduce_and_normalize(emb: np.ndarray, variance_threshold: float = 0.95,
                         n_components_fixed: int = None):
    logger.info("Fitting PCA ...")
    pca_full = PCA(n_components=min(emb.shape[0], emb.shape[1]), svd_solver="randomized",
                   random_state=42)
    pca_full.fit(emb)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    if n_components_fixed is not None:
        n_components = n_components_fixed
        logger.info(f"PCA: using fixed {n_components} components "
                    f"(explains {cumvar[n_components-1]:.1%} variance)")
    else:
        n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
        logger.info(f"PCA: {n_components} components explain {cumvar[n_components-1]:.1%} variance "
                    f"(threshold {variance_threshold:.0%})")

    # Fit PCA with the chosen number of components
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(emb)  # (N, n_components)

    # Save the explained variance plot
    return reduced, pca, cumvar, n_components


def plot_explained_variance(cumvar, n_components, out_path):
    # Truncate display to first 300 dims for readability, but always include n_components
    max_display = max(300, n_components + 20)
    plot_cumvar = cumvar[:min(max_display, len(cumvar))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(plot_cumvar) + 1), plot_cumvar, linewidth=1)
    ax.axvline(n_components, color="red", linestyle="--",
               label=f"{n_components} components → {cumvar[n_components-1]:.1%} variance")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA explained variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved PCA variance plot → {out_path}")


# ---------------------------------------------------------------------------
# Per-layer renormalization (for sparse optC/optD embeddings)
# ---------------------------------------------------------------------------

def renormalize_per_layer(emb: np.ndarray, num_layers: int = 16, num_experts: int = 127):
    """
    Renormalize each layer's block so values sum to 1.0 per layer.

    For sparse embeddings (optC/optD), the top-32 softmax probs may only sum to ~0.4
    per layer. This rescales them to proper distributions.
    """
    N = emb.shape[0]
    reshaped = emb.reshape(N, num_layers, num_experts)
    layer_sums = reshaped.sum(axis=2, keepdims=True)
    layer_sums = np.where(layer_sums == 0, 1.0, layer_sums)  # avoid div-by-zero
    reshaped = reshaped / layer_sums
    return reshaped.reshape(N, num_layers * num_experts)


# ---------------------------------------------------------------------------
# TruncatedSVD (sparsity-preserving dimensionality reduction)
# ---------------------------------------------------------------------------

def reduce_tsvd(emb: np.ndarray, variance_threshold: float = 0.95,
                n_components_fixed: int = None):
    """
    Like reduce_and_normalize() but uses TruncatedSVD instead of PCA.

    TruncatedSVD does NOT center the data, so zeros stay near zero — important
    for sparse embeddings where PCA's mean-centering destroys sparsity structure.
    """
    max_components = min(emb.shape[0], emb.shape[1]) - 1  # SVD can't do full rank
    logger.info(f"Fitting TruncatedSVD (max {max_components} components) ...")

    # First fit with many components to find explained variance curve
    n_fit = min(max_components, 500)
    svd_full = TruncatedSVD(n_components=n_fit, random_state=42)
    svd_full.fit(emb)

    cumvar = np.cumsum(svd_full.explained_variance_ratio_)
    if n_components_fixed is not None:
        n_components = min(n_components_fixed, n_fit)
        logger.info(f"TruncatedSVD: using fixed {n_components} components "
                    f"(explains {cumvar[n_components-1]:.1%} variance)")
    else:
        n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
        n_components = min(n_components, n_fit)
        logger.info(f"TruncatedSVD: {n_components} components explain "
                    f"{cumvar[n_components-1]:.1%} variance (threshold {variance_threshold:.0%})")

    # Fit with chosen components
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(emb)

    return reduced, cumvar, n_components


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def apply_transform(
    emb: np.ndarray,
    transform: str,
    variance_threshold: float = 0.95,
    n_components_fixed: int = None,
    num_layers: int = 16,
    num_experts: int = 127,
):
    """
    Apply a feature-engineering pipeline to raw embeddings before clustering.

    Args:
        emb: (N, D) float array of raw embeddings
        transform: one of raw | l2 | pca_l2 | log_l2 | standardize_pca_l2 |
                   renorm_l2 | tsvd_l2 | renorm_tsvd_l2
        variance_threshold: PCA/SVD explained-variance cutoff
        n_components_fixed: override PCA/SVD component count (overrides variance_threshold)
        num_layers: number of MoE layers (for per-layer renormalization)
        num_experts: number of experts per layer (for per-layer renormalization)

    Returns:
        transformed: (N, D') float32 array ready for K-means
        pca_info: (cumvar, n_components) tuple if PCA/SVD was applied, else None
    """
    emb = emb.astype(np.float32)

    if transform == "raw":
        return emb, None

    elif transform == "l2":
        return normalize(emb, norm="l2"), None

    elif transform == "pca_l2":
        reduced, _, cumvar, n_comps = reduce_and_normalize(emb, variance_threshold, n_components_fixed)
        return normalize(reduced, norm="l2"), (cumvar, n_comps)

    elif transform == "log_l2":
        return normalize(np.log1p(emb), norm="l2"), None

    elif transform == "standardize_pca_l2":
        scaled = StandardScaler().fit_transform(emb)
        reduced, _, cumvar, n_comps = reduce_and_normalize(scaled, variance_threshold, n_components_fixed)
        return normalize(reduced, norm="l2"), (cumvar, n_comps)

    elif transform == "renorm_l2":
        renormed = renormalize_per_layer(emb, num_layers, num_experts)
        return normalize(renormed, norm="l2"), None

    elif transform == "tsvd_l2":
        reduced, cumvar, n_comps = reduce_tsvd(emb, variance_threshold, n_components_fixed)
        return normalize(reduced, norm="l2"), (cumvar, n_comps)

    elif transform == "renorm_tsvd_l2":
        renormed = renormalize_per_layer(emb, num_layers, num_experts)
        reduced, cumvar, n_comps = reduce_tsvd(renormed, variance_threshold, n_components_fixed)
        return normalize(reduced, norm="l2"), (cumvar, n_comps)

    else:
        raise ValueError(f"Unknown transform: {transform!r}")


# ---------------------------------------------------------------------------
# K-means sweep
# ---------------------------------------------------------------------------

def run_sweep(reduced_normed: np.ndarray, k_values, output_dir: str):
    inertias, silhouettes = [], []

    for k in k_values:
        logger.info(f"K-means k={k} ...")
        km = MiniBatchKMeans(n_clusters=k, n_init=5, max_iter=300,
                             batch_size=4096, random_state=42)
        labels = km.fit_predict(reduced_normed)
        inertias.append(km.inertia_)

        # Silhouette on a subsample (expensive for large N)
        n_sample = min(5000, len(reduced_normed))
        idx = np.random.default_rng(42).choice(len(reduced_normed), n_sample, replace=False)
        sil = silhouette_score(reduced_normed[idx], labels[idx], metric="euclidean",
                               sample_size=None)
        silhouettes.append(sil)
        logger.info(f"  k={k:>4}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(k_values, inertias, "o-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow curve")
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, silhouettes, "o-", color="green")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette scores")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "kmeans_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved sweep plot → {plot_path}")

    # Save numbers
    sweep_path = os.path.join(output_dir, "kmeans_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({"k_values": k_values, "inertias": inertias, "silhouettes": silhouettes}, f, indent=2)

    logger.info("\n--- Sweep summary ---")
    for k, inertia, sil in zip(k_values, inertias, silhouettes):
        logger.info(f"  k={k:>4}  inertia={inertia:>12.1f}  silhouette={sil:.4f}")

    return inertias, silhouettes


# ---------------------------------------------------------------------------
# Final clustering + report
# ---------------------------------------------------------------------------

def run_final_cluster(reduced_normed: np.ndarray, emb_orig: np.ndarray,
                      meta: list, info: dict, k: int, output_dir: str,
                      num_layers: int = None, num_experts: int = None):
    logger.info(f"\nRunning final K-means with k={k} ...")
    km = MiniBatchKMeans(n_clusters=k, n_init=10, max_iter=500,
                         batch_size=4096, random_state=42)
    labels = km.fit_predict(reduced_normed)

    cluster_dir = os.path.join(output_dir, f"clusters_k{k}")
    os.makedirs(cluster_dir, exist_ok=True)

    # Save assignments
    np.save(os.path.join(cluster_dir, "assignments.npy"), labels)

    # Per-cluster analysis
    sources = [m["source"] for m in meta]
    unique_sources = sorted(set(sources))
    num_layers = num_layers if num_layers is not None else info["num_layers"]
    num_experts = num_experts if num_experts is not None else info["num_standard_experts"]

    report_lines = [f"Clustering report: k={k}, {len(labels)} documents\n"]
    cluster_summaries = []

    for c in range(k):
        mask = labels == c
        c_indices = np.where(mask)[0]
        c_size = mask.sum()

        if c_size == 0:
            cluster_summaries.append({
                "cluster": int(c), "size": 0, "source_counts": {},
                "top10_experts_global": [], "representative_docs": [],
            })
            report_lines.append(f"\n{'='*70}")
            report_lines.append(f"CLUSTER {c}  (0 docs, empty)")
            continue

        # Source breakdown
        c_sources = [sources[i] for i in c_indices]
        source_counts = {s: c_sources.count(s) for s in unique_sources if c_sources.count(s) > 0}
        source_str = "  ".join(f"{s}:{n}({n/c_size:.0%})" for s, n in
                               sorted(source_counts.items(), key=lambda x: -x[1]))

        # Top activated experts per layer (from original float32 embeddings, averaged over cluster)
        c_emb_mean = emb_orig[mask].mean(axis=0)  # (2032,)
        per_layer = c_emb_mean.reshape(num_layers, num_experts)  # (16, 127)
        # Global top-10 experts (summed across layers)
        global_expert_usage = per_layer.sum(axis=0)  # (127,)
        top10_experts = np.argsort(global_expert_usage)[::-1][:10].tolist()

        # Representative documents: 5 closest to centroid in PCA space
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(reduced_normed[mask] - centroid, axis=1)
        nearest_local = np.argsort(dists)[:5]
        nearest_global = c_indices[nearest_local]

        rep_docs = []
        for idx in nearest_global:
            rep_docs.append({
                "source": meta[idx]["source"],
                "doc_len": meta[idx]["doc_len"],
                "preview": meta[idx]["preview"],
            })

        cluster_summaries.append({
            "cluster": int(c),
            "size": int(c_size),
            "source_counts": {k2: int(v) for k2, v in source_counts.items()},
            "top10_experts_global": top10_experts,
            "representative_docs": rep_docs,
        })

        # Report text
        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"CLUSTER {c}  ({c_size} docs, {c_size/len(labels):.1%})")
        report_lines.append(f"Sources: {source_str}")
        report_lines.append(f"Top experts (summed across layers): {top10_experts}")
        report_lines.append("Representative documents:")
        for i, doc in enumerate(rep_docs):
            report_lines.append(f"  [{i+1}] ({doc['source']}, {doc['doc_len']} tokens)")
            report_lines.append(f"       {doc['preview'][:200]}")

    # Save JSON summary
    summary_path = os.path.join(cluster_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(cluster_summaries, f, indent=2)

    # Save text report
    report_path = os.path.join(cluster_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved assignments → {cluster_dir}/assignments.npy")
    logger.info(f"Saved summary     → {summary_path}")
    logger.info(f"Saved report      → {report_path}")

    # Print a condensed version to stdout
    logger.info(f"\n--- Cluster overview (k={k}) ---")
    for s in cluster_summaries:
        dominant = max(s["source_counts"], key=s["source_counts"].get)
        dom_pct = s["source_counts"][dominant] / s["size"]
        logger.info(f"  Cluster {s['cluster']:>3}: {s['size']:>5} docs | "
                    f"dominant={dominant}({dom_pct:.0%}) | "
                    f"top experts={s['top10_experts_global'][:5]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="claude_outputs/analysis/router_clustering")
    parser.add_argument("--mode", choices=["sweep", "cluster"], required=True,
                        help="sweep: run elbow/silhouette plots; cluster: run final clustering")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of clusters (required for --mode cluster)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32, 64, 128],
                        help="k values to sweep (default: 8 16 32 64 128)")
    parser.add_argument("--transform", default="pca_l2",
                        choices=["raw", "l2", "pca_l2", "log_l2", "standardize_pca_l2",
                                 "renorm_l2", "tsvd_l2", "renorm_tsvd_l2"],
                        help="Feature transform pipeline applied before K-means (default: pca_l2)")
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--n-components", type=int, default=None,
                        help="Fix PCA components directly (overrides --variance-threshold)")
    parser.add_argument("--emb-file", type=str, default=None,
                        help="Path to embedding .npy file (default: <data-dir>/embeddings_optA_avgprob.npy)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dir with shared data files (metadata.jsonl.gz, info.json). "
                             "Defaults to --output-dir if not specified.")
    parser.add_argument("--num-layers", type=int, default=16,
                        help="Number of MoE layers (for per-layer renormalization, default: 16)")
    parser.add_argument("--num-experts", type=int, default=127,
                        help="Number of experts per layer (for per-layer renormalization, default: 127)")
    args = parser.parse_args()

    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    emb, meta, info = load_data(args.output_dir, args.emb_file, args.data_dir)

    logger.info(f"Applying transform: {args.transform}")
    transformed, pca_info = apply_transform(
        emb, args.transform, args.variance_threshold, args.n_components,
        args.num_layers, args.num_experts,
    )

    if pca_info is not None:
        cumvar, n_components = pca_info
        plot_explained_variance(
            cumvar,
            n_components,
            os.path.join(args.output_dir, "pca_variance.png")
        )

    if args.mode == "sweep":
        run_sweep(transformed, args.k_values, args.output_dir)
        logger.info("\nReview kmeans_sweep.png then rerun with --mode cluster --k <chosen_k>")

    elif args.mode == "cluster":
        if args.k is None:
            raise ValueError("--k is required for --mode cluster")
        run_final_cluster(transformed, emb.astype(np.float32), meta, info,
                          args.k, args.output_dir,
                          num_layers=args.num_layers, num_experts=args.num_experts)


if __name__ == "__main__":
    main()
