"""
Characterize what makes each document-router cluster distinct.

For a saved k-way clustering of a document-level router embedding, answers: is each
cluster defined by *consistently routing to a few experts* (concentrated / few-expert),
or by *near-average usage with subtle, distributed differences* (distributed / subtle)?
Run this for both `doc_probs` (soft router affinity) and `doc_topk_freq` (hard expert
selection); the contrast between them is itself the answer.

Reuses the raw-(layer,expert)-space tests from cluster_expert_attribution.py
(signature_concentration, single_dim_auc, full_pattern_auc) and adds:
  - absolute centroid concentration (entropy of the cluster's mean routing vs global),
  - within-cluster consistency (cosine-to-centroid, CV + selection rate of signature experts),
  - a per-cluster concentrated/distributed verdict + an overall headline.

Outputs metrics.json + figures to <output-dir>. The driver
(scripts/modular_extension/characterize_clusters.sh) runs it per embedding and combines.

Usage:
    PYTHONPATH=.:src python -m src.scripts.clustering.characterize_cluster_expert_usage \\
        --data-dir modular_extension/cluster/emo100b_step23842 \\
        --embedding doc_probs --k 64 \\
        --output-dir modular_extension/cluster/emo100b_step23842/expert_attribution/doc_probs
"""

import argparse
import json
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.scripts.clustering.cluster_expert_attribution import (
    full_pattern_auc,
    signature_concentration,
    single_dim_auc,
)
from src.scripts.clustering.transform import load_embedding

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _entropy_nats(dist, axis=-1):
    """Shannon entropy in nats along `axis` for a distribution that sums to 1."""
    p = np.clip(dist, 1e-12, None)
    return -(dist * np.log(p)).sum(axis=axis)


def cluster_means_3d(emb, labels, k, num_layers, num_experts):
    """Per-cluster mean expert vector, reshaped to (k, num_layers, num_experts)."""
    D = num_layers * num_experts
    means = np.zeros((k, D), dtype=np.float64)
    for c in range(k):
        means[c] = emb[labels == c].mean(axis=0)
    return means.reshape(k, num_layers, num_experts)


def absolute_concentration(emb, labels, k, num_layers, num_experts):
    """Entropy of each cluster's mean routing distribution (per layer, avg over layers),
    vs the global mean. Low effective_experts / negative delta => the cluster concentrates
    its routing on fewer experts than the corpus average."""
    means3d = cluster_means_3d(emb, labels, k, num_layers, num_experts)
    # normalize each layer slice to a distribution (doc_probs~1 already; topk_freq sums to 7)
    means_dist = means3d / np.clip(means3d.sum(axis=-1, keepdims=True), 1e-12, None)
    H = _entropy_nats(means_dist, axis=-1).mean(axis=1)  # (k,) mean-over-layers entropy, nats

    gmean = emb.mean(axis=0).reshape(num_layers, num_experts)
    gmean = gmean / np.clip(gmean.sum(axis=-1, keepdims=True), 1e-12, None)
    Hg = float(_entropy_nats(gmean, axis=-1).mean())
    return {
        "effective_experts_used": np.exp(H),  # (k,)
        "entropy_nats": H,
        "global_effective_experts": float(np.exp(Hg)),
        "global_entropy_nats": Hg,
        "delta_entropy_vs_global": H - Hg,
        "max_experts": num_experts,
    }


def within_cluster_consistency(emb, labels, k, deltas, topk_emb, num_experts, top_n=5):
    """For each cluster: mean cosine sim of docs to the raw-space centroid, and for the
    top-`top_n` signature experts (by |delta|), their coefficient of variation across the
    cluster's docs and their selection rate (fraction of docs that route any token there,
    read from doc_topk_freq)."""
    from sklearn.preprocessing import normalize

    Xn = normalize(emb, norm="l2")
    rows = []
    for c in range(k):
        mask = labels == c
        sub = emb[mask]
        cen = normalize(sub.mean(axis=0, keepdims=True), norm="l2")[0]
        cos = float((Xn[mask] @ cen).mean())
        sig = np.argsort(-np.abs(deltas[c]))[:top_n]
        vals = sub[:, sig]  # (n_c, top_n)
        mean = vals.mean(axis=0)
        cv = (vals.std(axis=0) / np.clip(np.abs(mean), 1e-9, None)).tolist()
        sel = (topk_emb[mask][:, sig] > 0).mean(axis=0).tolist()  # selection rate
        rows.append(
            {
                "mean_cosine_to_centroid": cos,
                "signature_experts": [
                    {"layer": int(d) // num_experts, "expert": int(d) % num_experts,
                     "delta": float(deltas[c, d]), "cv_across_docs": float(cv[i]),
                     "selection_rate": float(sel[i])}
                    for i, d in enumerate(sig)
                ],
                "mean_signature_cv": float(np.mean(cv)),
            }
        )
    return rows


def verdict(effective_dims, best_auc, full_auc, n_dims):
    """Place each cluster on two independent axes:

      breadth  = how many experts carry the distinguishing signal
                 (effective_dims: few  <=10% of dims  vs  broad)
      strength = does a SINGLE expert nearly identify the cluster (strong),
                 or do you need the whole pattern (subtle)?
                 ratio = (best_single_auc - .5) / (full_pattern_auc - .5)

    'few-expert'      = a handful of experts, each strongly discriminative (sparse specialization)
    'broad-redundant' = many experts each shifted enough that any one identifies the cluster
    'subtle'          = many experts, none individually decisive; only the joint pattern separates
    """
    ratio = (best_auc - 0.5) / max(full_auc - 0.5, 1e-6)
    strong = best_auc >= 0.8 and ratio >= 0.7
    sparse = effective_dims <= 0.1 * n_dims
    if not strong:
        return "subtle", ratio
    return ("few-expert" if sparse else "broad-redundant"), ratio


# ── plots ────────────────────────────────────────────────────────────────────


def plot_concentration(rows, n_dims, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    clusters = sorted(rows, key=lambda r: -r["top5_mass"])
    xs = np.arange(len(clusters))
    for m, color in [(1, "#d62728"), (5, "#ff7f0e"), (10, "#2ca02c"), (32, "#1f77b4")]:
        ax.plot(xs, [r[f"top{m}_mass"] for r in clusters], marker=".", label=f"top-{m} dims", color=color)
    ax.set_xlabel("cluster (sorted by top-5 mass)")
    ax.set_ylabel("fraction of |centroid deviation| mass")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"How concentrated is each cluster's expert signature? ({n_dims} dims total)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_auc(best_auc, full_auc, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = np.argsort(-best_auc)
    xs = np.arange(len(order))
    ax.plot(xs, full_auc[order], marker=".", label="full pattern (cos-sim to centroid)", color="#1f77b4")
    ax.plot(xs, best_auc[order], marker=".", label="best single (layer, expert)", color="#d62728")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("cluster (sorted by best single-dim AUC)")
    ax.set_ylabel("one-vs-rest AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend()
    ax.set_title("Cluster separability: single best expert vs full activation pattern")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_deviation_heatmap(deltas, global_std, num_layers, num_experts, order, out):
    z = deltas / np.clip(global_std, 1e-9, None)  # (k, D) standardized mean difference
    z = z[order]
    lim = np.percentile(np.abs(z), 99)
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    for l in range(1, num_layers):
        ax.axvline(l * num_experts - 0.5, color="k", lw=0.3, alpha=0.4)
    ax.set_xticks([(l + 0.5) * num_experts for l in range(num_layers)])
    ax.set_xticklabels([f"L{l}" for l in range(num_layers)], fontsize=8)
    ax.set_xlabel("layer × expert (63 experts per layer)")
    ax.set_ylabel("cluster (sorted by concentration)")
    ax.set_title("Per-cluster expert-usage deviation from global mean (z-scored)")
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="std devs above/below global mean")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_verdict_scatter(rows, n_dims, out):
    """Two axes of the user's question: breadth (how many experts carry the signal) x
    strength (can a single expert identify the cluster, or only the whole pattern)."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = {"few-expert": "#d62728", "broad-redundant": "#9467bd", "subtle": "#1f77b4"}
    for v, col in colors.items():
        r = [x for x in rows if x["verdict"] == v]
        if not r:
            continue
        ax.scatter(
            [x["effective_dims"] for x in r],
            [x["single_vs_full_ratio"] for x in r],
            s=[max(20, x["size"] / 120) for x in r],
            c=col, alpha=0.7, edgecolors="k", linewidths=0.4, label=v,
        )
    ax.axvline(0.1 * n_dims, color="gray", ls="--", lw=1)
    ax.axhline(0.7, color="gray", ls="--", lw=1)
    ax.text(0.1 * n_dims + 6, 0.72, "← few experts | many experts →", color="#888", fontsize=8)
    ax.set_xlabel("effective # experts carrying the deviation  (few  ←→  broad)")
    ax.set_ylabel("single-expert / full-pattern separability  (subtle ↓  |  strong ↑)")
    ax.set_title("What defines each cluster? (point size = # docs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_example_profiles(means3d, rows, num_layers, num_experts, out):
    ordering = sorted(range(len(rows)), key=lambda c: rows[c]["effective_dims"])
    picks = ordering[:4] + ordering[-4:]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    vmax = np.percentile(means3d, 99)
    for ax, c in zip(axes.ravel(), picks):
        im = ax.imshow(means3d[c].T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(f"cluster {c} ({rows[c]['verdict']}, eff={rows[c]['effective_dims']:.0f})", fontsize=9)
        ax.set_xlabel("layer", fontsize=8)
        ax.set_ylabel("expert", fontsize=8)
        ax.set_xticks(range(0, num_layers, 4))
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label="mean routing to expert")
    fig.suptitle("Example per-cluster expert profiles (most concentrated → most distributed)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--embedding", default="doc_probs", help="feature view to CHARACTERIZE")
    p.add_argument("--cluster-embedding", default="doc_probs",
                   help="embedding the clustering was FIT on (locates assignments.npy); the same "
                        "cluster labels are analyzed under whichever --embedding view is chosen")
    p.add_argument("--preprocess", default="mean_pca_l2")
    p.add_argument("--method", default="spherical_kmeans")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cluster_dir = os.path.join(
        args.data_dir, f"{args.cluster_embedding}_{args.preprocess}_{args.method}_k{args.k}")
    labels = np.load(os.path.join(cluster_dir, "assignments.npy"))
    emb, meta, info = load_embedding(args.data_dir, args.embedding)
    NE, NL, k = info["num_standard_experts"], info["num_layers"], args.k
    topics = np.array([m["source"] for m in meta])
    logger.info(f"emb={emb.shape} k={k} layers={NL} experts/layer={NE} embedding={args.embedding}")

    logger.info("signature concentration")
    conc = signature_concentration(emb, labels, k)
    deltas = conc["deltas"]

    logger.info("single-dim AUC")
    auc = single_dim_auc(emb, labels, k)  # (D, k)
    auc_sym = np.maximum(auc, 1.0 - auc)
    best_dim, best_auc = auc_sym.argmax(axis=0), auc_sym.max(axis=0)
    emb_centered = emb - emb.mean(axis=0)
    full_auc = full_pattern_auc(emb_centered, labels, k)  # raw expert space, apples-to-apples
    pca_path = os.path.join(args.data_dir, f"preprocessed_{args.cluster_embedding}_{args.preprocess}.npy")
    if args.embedding == args.cluster_embedding and os.path.exists(pca_path):
        full_auc = full_pattern_auc(np.load(pca_path), labels, k)  # space the clustering was fit in

    logger.info("absolute concentration + consistency")
    absc = absolute_concentration(emb, labels, k, NL, NE)
    means3d = cluster_means_3d(emb, labels, k, NL, NE)
    topk_emb, _, _ = load_embedding(args.data_dir, "doc_topk_freq")
    cons = within_cluster_consistency(emb, labels, k, deltas, topk_emb, NE)
    global_std = emb.std(axis=0)

    per_cluster = []
    for c in range(k):
        cr = dict(conc["per_cluster"][c])
        d = int(best_dim[c])
        v, ratio = verdict(cr["effective_dims"], float(best_auc[c]), float(full_auc[c]), emb.shape[1])
        t_counts: dict = {}
        for t in topics[labels == c]:
            t_counts[t] = t_counts.get(t, 0) + 1
        cr.update(
            {
                "best_dim": {"layer": d // NE, "expert": d % NE, "auc": float(best_auc[c]),
                             "direction": "over" if auc[d, c] >= 0.5 else "under"},
                "full_pattern_auc": float(full_auc[c]),
                "single_vs_full_ratio": float(ratio),
                "effective_experts_used": float(absc["effective_experts_used"][c]),
                "delta_entropy_vs_global": float(absc["delta_entropy_vs_global"][c]),
                "mean_cosine_to_centroid": cons[c]["mean_cosine_to_centroid"],
                "mean_signature_cv": cons[c]["mean_signature_cv"],
                "signature_experts": cons[c]["signature_experts"],
                "top_topics": sorted(t_counts.items(), key=lambda x: -x[1])[:3],
                "dominant_topic": max(t_counts, key=t_counts.get),
                "verdict": v,
            }
        )
        per_cluster.append(cr)

    total = len(labels)
    VERDICTS = ("few-expert", "broad-redundant", "subtle")
    counts = {v: sum(r["verdict"] == v for r in per_cluster) for v in VERDICTS}
    doc_frac = {
        v: float(sum(r["size"] for r in per_cluster if r["verdict"] == v) / total) for v in VERDICTS
    }
    summary = {
        "data_dir": args.data_dir, "embedding": args.embedding, "k": k, "n_dims": int(emb.shape[1]),
        "n_layers": NL, "n_experts_per_layer": NE,
        "global_effective_experts": absc["global_effective_experts"], "max_experts": NE,
        "verdict_counts": counts,
        "verdict_doc_fraction": doc_frac,
        "median_effective_dims": float(np.median([r["effective_dims"] for r in per_cluster])),
        "median_top5_mass": float(np.median([r["top5_mass"] for r in per_cluster])),
        "median_best_single_dim_auc": float(np.median(best_auc)),
        "median_full_pattern_auc": float(np.median(full_auc)),
        "median_single_vs_full_ratio": float(np.median([r["single_vs_full_ratio"] for r in per_cluster])),
        "median_effective_experts_used": float(np.median(absc["effective_experts_used"])),
        "median_cosine_to_centroid": float(np.median([r["mean_cosine_to_centroid"] for r in per_cluster])),
        "per_cluster": per_cluster,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    order = np.argsort([r["effective_dims"] for r in per_cluster])
    plot_concentration(conc["per_cluster"], emb.shape[1], os.path.join(args.output_dir, "signature_concentration.png"))
    plot_auc(best_auc, full_auc, os.path.join(args.output_dir, "single_dim_vs_full_auc.png"))
    plot_deviation_heatmap(deltas, global_std, NL, NE, order, os.path.join(args.output_dir, "deviation_heatmap.png"))
    plot_verdict_scatter(per_cluster, emb.shape[1], os.path.join(args.output_dir, "verdict_scatter.png"))
    plot_example_profiles(means3d, per_cluster, NL, NE, os.path.join(args.output_dir, "example_profiles.png"))

    logger.info("--- headline (%s) ---", args.embedding)
    logger.info("  verdict: %d few-expert / %d broad-redundant / %d subtle",
                counts["few-expert"], counts["broad-redundant"], counts["subtle"])
    logger.info("  median effective deviation dims: %.1f / %d", summary["median_effective_dims"], emb.shape[1])
    logger.info("  median best-single vs full-pattern AUC: %.3f vs %.3f",
                summary["median_best_single_dim_auc"], summary["median_full_pattern_auc"])
    logger.info("  median effective experts USED/layer: %.1f / %d (global %.1f)",
                summary["median_effective_experts_used"], NE, absc["global_effective_experts"])
    logger.info("Saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
