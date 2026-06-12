"""
Are router-prob clusters driven by individual experts or by broad activation
patterns?

Given a doc-level embedding (e.g. doc_probs) and a saved clustering of it,
runs three tests in the raw (layer, expert) dimension space:

1. Signature concentration: for each cluster, how concentrated is the
   centroid's deviation from the global mean across the layer*expert dims?
   (top-m mass fractions + effective number of dims)

2. Single-dim separability: for each cluster, the best single (layer, expert)
   dim's AUC at separating the cluster from the rest, vs the full-pattern
   baseline (cosine similarity to the cluster centroid).

3. Necessity / sufficiency ablations: drop (or keep only) the union of each
   cluster's top-m dims, re-run the full preprocess + clustering pipeline,
   and measure agreement with the original clustering (ARI) and with the
   weborganizer topic labels (NMI). Random-dim drops of matched size give the
   pipeline-instability baseline.

Usage:
    python -m src.scripts.clustering.cluster_expert_attribution \
        --data-dir claude_outputs/models_sizescaling/weborganizer/emo_1b14b_130b \
        --embedding doc_probs --preprocess mean_pca_l2 \
        --method spherical_kmeans --k 32 \
        --output-dir claude_outputs/models_sizescaling/expert_attribution/emo_1b14b_130b
"""

import argparse
import json
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.scripts.clustering.cluster import CLUSTER_REGISTRY
from src.scripts.clustering.transform import apply_preprocess, load_embedding

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test 1: signature concentration
# ---------------------------------------------------------------------------


def signature_concentration(emb: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """Per-cluster concentration of |centroid - global mean| over raw dims."""
    global_mean = emb.mean(axis=0)
    top_ms = [1, 5, 10, 32, 100]
    rows = []
    deltas = np.zeros((k, emb.shape[1]), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        delta = emb[mask].mean(axis=0) - global_mean
        deltas[c] = delta
        w = np.abs(delta)
        w_sorted = np.sort(w)[::-1]
        total = w_sorted.sum()
        p = w_sorted / total
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        rows.append(
            {
                "cluster": int(c),
                "size": int(mask.sum()),
                **{f"top{m}_mass": float(w_sorted[:m].sum() / total) for m in top_ms},
                "effective_dims": float(np.exp(ent)),
            }
        )
    return {"per_cluster": rows, "deltas": deltas}


# ---------------------------------------------------------------------------
# Test 2: single-dim AUC vs full-pattern AUC
# ---------------------------------------------------------------------------


def single_dim_auc(emb: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """AUC of every raw dim for every one-vs-rest cluster task. Returns (D, k).

    Mann-Whitney AUC from column-wise ranks: AUC = (R_pos - n_pos(n_pos+1)/2)
    / (n_pos * n_neg), computed for all dims x clusters at once.
    """
    from scipy.stats import rankdata

    N, D = emb.shape
    ranks = rankdata(emb, axis=0).astype(np.float64)  # (N, D)
    onehot = np.zeros((N, k), dtype=np.float64)
    onehot[np.arange(N), labels] = 1.0
    n_pos = onehot.sum(axis=0)  # (k,)
    rank_sums = ranks.T @ onehot  # (D, k)
    auc = (rank_sums - n_pos * (n_pos + 1) / 2.0) / (n_pos * (N - n_pos))
    return auc


def full_pattern_auc(transformed: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """AUC of cosine-sim-to-own-centroid score for each cluster (baseline)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import normalize

    X = normalize(transformed, norm="l2")
    aucs = np.zeros(k)
    for c in range(k):
        mask = labels == c
        centroid = normalize(X[mask].mean(axis=0, keepdims=True), norm="l2")[0]
        score = X @ centroid
        aucs[c] = roc_auc_score(mask.astype(int), score)
    return aucs


# ---------------------------------------------------------------------------
# Test 3: drop / keep ablations
# ---------------------------------------------------------------------------


def recluster(
    emb: np.ndarray, dims_mask: np.ndarray, preprocess: str, method: str, k: int, info: dict
) -> np.ndarray:
    """Re-run the full pipeline on a dim-subset of the raw embedding."""
    sub = emb[:, dims_mask]
    transformed = apply_preprocess(sub, preprocess, info).astype(np.float32)
    return CLUSTER_REGISTRY[method]["fn"](transformed, k)


def top_dims_union(deltas: np.ndarray, m: int) -> np.ndarray:
    """Union over clusters of each cluster's top-m |delta| dims."""
    idx = np.argsort(-np.abs(deltas), axis=1)[:, :m]
    return np.unique(idx)


def run_ablations(
    emb: np.ndarray,
    labels: np.ndarray,
    topics: np.ndarray,
    deltas: np.ndarray,
    preprocess: str,
    method: str,
    k: int,
    info: dict,
    drop_ms: list,
    keep_ms: list,
) -> list:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    D = emb.shape[1]
    rng = np.random.default_rng(0)
    results = []

    def record(kind, m, dims_removed_or_kept, new_labels):
        results.append(
            {
                "kind": kind,
                "m": int(m),
                "n_dims_used": int(D - dims_removed_or_kept if kind.startswith("drop") else dims_removed_or_kept),
                "ari_vs_original": float(adjusted_rand_score(labels, new_labels)),
                "nmi_vs_original": float(normalized_mutual_info_score(labels, new_labels)),
                "nmi_vs_topics": float(normalized_mutual_info_score(topics, new_labels)),
            }
        )
        logger.info(
            f"  [{kind} m={m}] dims_used={results[-1]['n_dims_used']} "
            f"ARI={results[-1]['ari_vs_original']:.3f} "
            f"NMI_topics={results[-1]['nmi_vs_topics']:.3f}"
        )

    # Baseline: full pipeline, no ablation (captures k-means rerun stability).
    base_labels = recluster(emb, np.ones(D, dtype=bool), preprocess, method, k, info)
    record("baseline", 0, 0, base_labels)

    for m in drop_ms:
        union = top_dims_union(deltas, m)
        mask = np.ones(D, dtype=bool)
        mask[union] = False
        new_labels = recluster(emb, mask, preprocess, method, k, info)
        record("drop_top", m, len(union), new_labels)

        # Matched random control
        rand_dims = rng.choice(D, size=len(union), replace=False)
        mask_r = np.ones(D, dtype=bool)
        mask_r[rand_dims] = False
        new_labels_r = recluster(emb, mask_r, preprocess, method, k, info)
        record("drop_random", m, len(rand_dims), new_labels_r)

    for m in keep_ms:
        union = top_dims_union(deltas, m)
        mask = np.zeros(D, dtype=bool)
        mask[union] = True
        new_labels = recluster(emb, mask, preprocess, method, k, info)
        record("keep_top", m, len(union), new_labels)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_concentration(conc_rows: list, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    clusters = sorted(conc_rows, key=lambda r: -r["top5_mass"])
    xs = np.arange(len(clusters))
    for m, color in [(1, "#d62728"), (5, "#ff7f0e"), (10, "#2ca02c"), (32, "#1f77b4")]:
        ax.plot(xs, [r[f"top{m}_mass"] for r in clusters], marker=".", label=f"top-{m} dims", color=color)
    ax.set_xlabel("cluster (sorted by top-5 mass)")
    ax.set_ylabel("fraction of |centroid deviation| mass")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("How concentrated is each cluster's expert signature? (2032 dims total)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "signature_concentration.png"), dpi=150)
    plt.close(fig)


def plot_auc(best_dim_auc: np.ndarray, full_auc: np.ndarray, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = np.argsort(-best_dim_auc)
    xs = np.arange(len(order))
    ax.plot(xs, full_auc[order], marker=".", label="full pattern (cos-sim to centroid)", color="#1f77b4")
    ax.plot(xs, best_dim_auc[order], marker=".", label="best single (layer, expert) dim", color="#d62728")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("cluster (sorted by best single-dim AUC)")
    ax.set_ylabel("one-vs-rest AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend()
    ax.set_title("Cluster separability: single best expert vs full activation pattern")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "single_dim_vs_full_auc.png"), dpi=150)
    plt.close(fig)


def plot_ablations(results: list, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    base = next(r for r in results if r["kind"] == "baseline")
    for ax, metric, title in [
        (axes[0], "ari_vs_original", "agreement with original clusters (ARI)"),
        (axes[1], "nmi_vs_topics", "alignment with weborganizer topics (NMI)"),
    ]:
        for kind, label, color in [
            ("drop_top", "drop top-m dims/cluster", "#d62728"),
            ("drop_random", "drop matched random dims", "#7f7f7f"),
            ("keep_top", "keep ONLY top-m dims/cluster", "#2ca02c"),
        ]:
            rows = [r for r in results if r["kind"] == kind]
            ax.plot(
                [r["m"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                label=label,
                color=color,
            )
        ax.axhline(base[metric], color="#1f77b4", ls="--", lw=1, label="rerun baseline (no ablation)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m (top dims per cluster)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Do clusters survive removing their strongest experts?")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_curves.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embedding", default="doc_probs")
    parser.add_argument("--preprocess", default="mean_pca_l2")
    parser.add_argument("--method", default="spherical_kmeans")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--drop-ms", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--keep-ms", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cluster_dir = os.path.join(
        args.data_dir, f"{args.embedding}_{args.preprocess}_{args.method}_k{args.k}"
    )
    labels = np.load(os.path.join(cluster_dir, "assignments.npy"))
    emb, meta, info = load_embedding(args.data_dir, args.embedding)
    topics = np.array([m["source"] for m in meta])
    num_experts = info["num_standard_experts"]
    logger.info(f"emb={emb.shape}, k={args.k}, experts/layer={num_experts}")

    # --- Test 1: concentration
    logger.info("Test 1: signature concentration")
    conc = signature_concentration(emb, labels, args.k)
    deltas = conc["deltas"]

    # --- Test 2: single-dim AUC
    logger.info("Test 2: single-dim AUC (all dims x all clusters)")
    auc = single_dim_auc(emb, labels, args.k)  # (D, k)
    # An under-activated expert is also a marker: use max(auc, 1-auc).
    auc_sym = np.maximum(auc, 1.0 - auc)
    best_dim = auc_sym.argmax(axis=0)  # (k,)
    best_dim_auc = auc_sym.max(axis=0)
    transformed = np.load(
        os.path.join(args.data_dir, f"preprocessed_{args.embedding}_{args.preprocess}.npy")
    )
    full_auc = full_pattern_auc(transformed, labels, args.k)

    # --- Test 3: ablations
    logger.info("Test 3: drop/keep ablations")
    ablations = run_ablations(
        emb, labels, topics, deltas, args.preprocess, args.method, args.k, info,
        args.drop_ms, args.keep_ms,
    )

    # --- Per-cluster summary
    per_cluster = []
    for c in range(args.k):
        row = dict(conc["per_cluster"][c])
        d = int(best_dim[c])
        row["best_dim"] = {
            "layer": d // num_experts,
            "expert": d % num_experts,
            "auc": float(best_dim_auc[c]),
            "direction": "over" if auc[d, c] >= 0.5 else "under",
        }
        row["full_pattern_auc"] = float(full_auc[c])
        # Top-5 over-activated dims by delta, human-readable
        top5 = np.argsort(-deltas[c])[:5]
        row["top5_over_dims"] = [
            {"layer": int(t) // num_experts, "expert": int(t) % num_experts,
             "delta": float(deltas[c, t])}
            for t in top5
        ]
        # Dominant topics
        t_counts: dict = {}
        for t in topics[labels == c]:
            t_counts[t] = t_counts.get(t, 0) + 1
        row["top_topics"] = sorted(t_counts.items(), key=lambda x: -x[1])[:3]
        per_cluster.append(row)

    summary = {
        "data_dir": args.data_dir,
        "embedding": args.embedding,
        "k": args.k,
        "n_dims": int(emb.shape[1]),
        "concentration_mean": {
            f"top{m}_mass": float(np.mean([r[f"top{m}_mass"] for r in conc["per_cluster"]]))
            for m in [1, 5, 10, 32, 100]
        },
        "effective_dims_median": float(
            np.median([r["effective_dims"] for r in conc["per_cluster"]])
        ),
        "best_single_dim_auc": {
            "median": float(np.median(best_dim_auc)),
            "min": float(best_dim_auc.min()),
            "max": float(best_dim_auc.max()),
        },
        "full_pattern_auc_median": float(np.median(full_auc)),
        "ablations": ablations,
        "per_cluster": per_cluster,
    }
    with open(os.path.join(args.output_dir, "attribution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_concentration(conc["per_cluster"], args.output_dir)
    plot_auc(best_dim_auc, full_auc, args.output_dir)
    plot_ablations(ablations, args.output_dir)

    logger.info("--- Headline numbers ---")
    logger.info(f"  mean top-5 |delta| mass:    {summary['concentration_mean']['top5_mass']:.3f}")
    logger.info(f"  median effective dims:      {summary['effective_dims_median']:.1f} / {emb.shape[1]}")
    logger.info(f"  median best single-dim AUC: {summary['best_single_dim_auc']['median']:.3f}")
    logger.info(f"  median full-pattern AUC:    {summary['full_pattern_auc_median']:.3f}")
    logger.info(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
