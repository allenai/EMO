"""
Cluster document router embeddings to discover implicit domains learned by the MoE model.

Steps:
  1. Load Option A embeddings (avg softmax probs, float16)
  2. PCA to capture 95% variance
  3. L2 normalize (cosine-like distance for K-means)
  4. K-means sweep over k values → elbow + silhouette plots
  5. For chosen k: save assignments + per-cluster report

Usage:
    # Step 1: sweep to pick k (produces plots)
    python -m src.scripts.analysis.cluster_embeddings \
        --output-dir claude_outputs/analysis/router_clustering \
        --mode sweep

    # Step 2: run final clustering with chosen k
    python -m src.scripts.analysis.cluster_embeddings \
        --output-dir claude_outputs/analysis/router_clustering \
        --mode cluster --k 32
"""

import argparse
import gzip
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, linewidth=1)
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
                      meta: list, info: dict, k: int, output_dir: str):
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
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]

    report_lines = [f"Clustering report: k={k}, {len(labels)} documents\n"]
    cluster_summaries = []

    for c in range(k):
        mask = labels == c
        c_indices = np.where(mask)[0]
        c_size = mask.sum()

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
                "preview": meta[idx]["preview"][:300],
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
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--n-components", type=int, default=None,
                        help="Fix PCA components directly (overrides --variance-threshold)")
    parser.add_argument("--emb-file", type=str, default=None,
                        help="Path to embedding .npy file (default: <data-dir>/embeddings_optA_avgprob.npy)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dir with shared data files (metadata.jsonl.gz, info.json). "
                             "Defaults to --output-dir if not specified.")
    args = parser.parse_args()

    np.random.seed(42)

    emb, meta, info = load_data(args.output_dir, args.emb_file, args.data_dir)

    reduced, pca, cumvar, n_components = reduce_and_normalize(
        emb, args.variance_threshold, args.n_components)
    plot_explained_variance(
        cumvar[:min(300, len(cumvar))],  # plot first 300 dims for readability
        n_components,
        os.path.join(args.output_dir, "pca_variance.png")
    )

    # L2 normalize for cosine-like K-means
    reduced_normed = normalize(reduced, norm="l2")

    if args.mode == "sweep":
        run_sweep(reduced_normed, args.k_values, args.output_dir)
        logger.info("\nReview kmeans_sweep.png then rerun with --mode cluster --k <chosen_k>")

    elif args.mode == "cluster":
        if args.k is None:
            raise ValueError("--k is required for --mode cluster")
        run_final_cluster(reduced_normed, emb.astype(np.float32), meta, info,
                          args.k, args.output_dir)


if __name__ == "__main__":
    main()
