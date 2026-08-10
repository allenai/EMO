#!/usr/bin/env python3
"""Is a high-AUC expert cluster-specific? Sharing + spectrum illustrations.

Complements auc_illustration.py (which shows ONE cluster's metrics) with the
cross-cluster view, from the full separability matrix S (64 clusters x 1008
experts, one-vs-rest, computed exactly on all docs and cached):

  expert_attribution/one_expert_shared.png    -- panel-(A)-style in/out histograms
      of the SAME expert against each cluster it separates at >=0.9 (the most
      widely-shared expert). Each panel notes the out-pile share held by the
      OTHER leaning clusters: because clusters are small, they cost a few AUC
      points, not fifty -- that is how one expert has high AUC on many clusters.
  expert_attribution/best_expert_spectrum.png -- best-expert in/out histograms for
      12 clusters spanning the strong-separator-count spectrum (fewest ...
      median ... most=code), so Finding 2 is not illustrated by the code
      cluster alone.
  expert_attribution/expert_sharing.json      -- the numbers the report cites
      (sharing stats, per-cluster strong counts, usage amplification).
  expert_attribution/separability_matrix.npy  -- cached S (recomputed if absent;
      ~10 min of rankdata on the 1.15M x 1008 embeddings).

    python scripts/modular_extension/expert_sharing_illustration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1].parent
BASE = ROOT / "modular_extension/cluster/emo100b_step23842"
OUT = BASE / "expert_attribution"
N_EXP = 63  # standard experts per layer (layer-major: dim d -> layer d//63, expert d%63)
STRONG = 0.9


def separability_matrix(emb, labels, k=64) -> np.ndarray:
    """Exact one-vs-rest separability |AUC-0.5|+0.5 for every (cluster, dim).

    Rank each dim once over ALL docs; each cluster's rank-sum then gives its
    Mann-Whitney AUC vs the rest (ties handled by average ranks).
    """
    cache = OUT / "separability_matrix.npy"
    if cache.exists():
        return np.load(cache)
    n = len(labels)
    n_in = np.bincount(labels, minlength=k).astype(np.float64)
    n_out = n - n_in
    min_rank_sum = n_in * (n_in + 1) / 2.0
    S = np.zeros((k, emb.shape[1]))
    for j in range(emb.shape[1]):
        r = rankdata(np.asarray(emb[:, j], dtype=np.float32))
        auc = (np.bincount(labels, weights=r, minlength=k) - min_rank_sum) / (n_in * n_out)
        S[:, j] = np.abs(auc - 0.5) + 0.5
        if (j + 1) % 100 == 0:
            print(f"  ranked {j + 1}/{emb.shape[1]}")
    np.save(cache, S)
    return S


def in_out_hist(ax, x_in, x_out, color="#2563eb", hi=None):
    hi = hi or float(np.percentile(np.concatenate([x_in, x_out]), 99.5)) or 1e-4
    bins = np.linspace(0.0, hi, 60)
    ax.hist(x_out, bins=bins, density=True, alpha=0.55, color="#94a3b8")
    ax.hist(x_in, bins=bins, density=True, alpha=0.75, color=color)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)


def main():
    emb = np.load(BASE / "embeddings_doc_probs.npy", mmap_mode="r")
    labels = np.load(BASE / "doc_probs_mean_pca_l2_spherical_kmeans_k64/assignments.npy")
    topk = np.load(BASE / "embeddings_doc_topk_freq.npy", mmap_mode="r")
    N = len(labels)
    sizes = np.bincount(labels, minlength=64)
    rng = np.random.default_rng(0)

    S = separability_matrix(emb, labels)
    strong = S >= STRONG
    per_dim = strong.sum(axis=0)  # clusters served per dim
    per_cluster = strong.sum(axis=1)  # strong dims per cluster
    gmean = emb.mean(0, dtype=np.float64)

    # ---------------- figure 1: the most widely shared strong expert ----------
    dim = int(per_dim.argmax())
    cl = np.where(strong[:, dim])[0]
    cl = cl[np.argsort(-S[cl, dim])]
    col = np.asarray(emb[:, dim], dtype=np.float32)
    hi = float(np.percentile(col, 99.9))
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706",
              "#dc2626", "#0e7490", "#be185d", "#4d7c0f"]
    ncl = len(cl)
    ncols = (ncl + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 7.6))
    out_shares = {}
    for ax, c, color in zip(axes.flat, cl, colors):
        mask = labels == c
        out_idx = rng.choice(np.where(~mask)[0], size=100_000, replace=False)
        in_out_hist(ax, col[mask], col[out_idx], color=color, hi=hi)
        share = sizes[[k for k in cl if k != c]].sum() / (N - sizes[c])
        out_shares[int(c)] = round(float(share), 4)
        direction = "over" if col[mask].mean() > gmean[dim] else "under"
        ax.set_title(
            f"cluster {c} vs rest  ({direction})\n"
            f"sep AUC {S[c, dim]:.3f} — other {ncl - 1} leaning clusters\n"
            f"are only {share:.1%} of its out-pile",
            fontsize=9.5,
        )
    for ax in axes.flat[ncl:]:
        ax.axis("off")
    fig.suptitle(
        f"The SAME expert (L{dim // N_EXP}·E{dim % N_EXP}) separates {ncl} clusters at "
        f"≥{STRONG} — each is scored one-vs-REST,\nand the other leaning clusters are a "
        "tiny fraction of 'rest': they cost a few AUC points, not fifty.",
        fontsize=12,
    )
    fig.supxlabel(f"routing probability on L{dim // N_EXP}·E{dim % N_EXP}", fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.90))
    fig.savefig(OUT / "one_expert_shared.png", dpi=130)
    plt.close(fig)
    print("wrote", OUT / "one_expert_shared.png")

    # ---------------- figure 2: best expert across the redundancy spectrum ----
    picks = np.argsort(per_cluster)[np.linspace(0, 63, 12).round().astype(int)]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, c in zip(axes.flat, picks):
        d = int(S[c].argmax())
        mask = labels == c
        x_in = np.asarray(emb[mask, d], dtype=np.float32)
        out_idx = rng.choice(np.where(~mask)[0], size=100_000, replace=False)
        x_out = np.asarray(emb[out_idx, d], dtype=np.float32)
        in_out_hist(ax, x_in, x_out)
        delta = float(x_in.mean()) - gmean[d]
        ax.set_title(
            f"cluster {c}  ({sizes[c]:,} docs, {per_cluster[c]} experts ≥{STRONG})\n"
            f"best: L{d // N_EXP}·E{d % N_EXP}  sep {S[c, d]:.3f},  "
            f"|δ|={abs(delta):.3f},  σ={x_in.std():.3f}",
            fontsize=9.5,
        )
    axes.flat[0].legend(["out-pile (rest)", "in-pile (cluster)"], fontsize=9)
    fig.suptitle(
        "Best-expert in/out histograms for 12 clusters spanning the redundancy spectrum "
        f"(fewest → most experts ≥{STRONG}).\nEvery cluster — not just code — has a "
        "near-perfect single-expert separator.",
        fontsize=13,
    )
    fig.supxlabel("routing probability on the cluster's best expert", fontsize=11)
    fig.supylabel("density", fontsize=11)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.93))
    fig.savefig(OUT / "best_expert_spectrum.png", dpi=130)
    plt.close(fig)
    print("wrote", OUT / "best_expert_spectrum.png")

    # ---------------- stats the report cites ----------------------------------
    best = S.argmax(axis=1)
    mean_in = np.zeros((64, emb.shape[1]))
    mean_in_hard = np.zeros_like(mean_in)
    gmean_hard = topk.mean(0, dtype=np.float64)
    for c in range(64):
        mean_in[c] = emb[labels == c].mean(0, dtype=np.float64)
        mean_in_hard[c] = topk[labels == c].mean(0, dtype=np.float64)
    mass_share, n_other_50 = [], []
    for c in range(64):
        j = int(best[c])
        mass_share.append(sizes[c] * mean_in[c, j] / (N * gmean[j]))
        n_other_50.append(
            sum(mean_in[k, j] >= 0.5 * mean_in[c, j] for k in range(64) if k != c)
        )
    doc_share = sizes / N
    served = per_dim[per_dim > 0]
    stats = {
        "strong_threshold": STRONG,
        "sharing": {
            "n_dims_strong_for_someone": int((per_dim > 0).sum()),
            "n_dims": int(emb.shape[1]),
            "median_clusters_per_strong_dim": float(np.median(served)),
            "max_clusters_per_strong_dim": int(served.max()),
            "frac_strong_dims_shared_by_ge2": round(float((served >= 2).mean()), 3),
        },
        "per_cluster_strong_dims": {
            "median": float(np.median(per_cluster)),
            "min": int(per_cluster.min()),
            "max": int(per_cluster.max()),
            "argmax_cluster": int(per_cluster.argmax()),
        },
        "shared_expert_example": {
            "dim": dim, "layer": dim // N_EXP, "expert": dim % N_EXP,
            "n_clusters_ge_strong": ncl,
            "max_other_leaners_out_pile_share": max(out_shares.values()),
        },
        "best_expert_usage": {
            "median_share_of_experts_mass_from_cluster": round(float(np.median(mass_share)), 3),
            "median_cluster_doc_share": round(float(np.median(doc_share)), 4),
            "median_amplification": round(
                float(np.median(np.array(mass_share) / doc_share)), 1),
            "median_n_other_clusters_using_ge_50pct": float(np.median(n_other_50)),
            "median_hard_selection_in_cluster": round(
                float(np.median([mean_in_hard[c, best[c]] for c in range(64)])), 3),
            "median_hard_selection_global": round(
                float(np.median([gmean_hard[best[c]] for c in range(64)])), 3),
        },
        "spectrum_clusters": [int(c) for c in picks],
    }
    (OUT / "expert_sharing.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
