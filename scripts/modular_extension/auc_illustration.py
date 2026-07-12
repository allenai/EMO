#!/usr/bin/env python3
"""Illustrate the two per-cluster metrics on one concrete cluster.

Produces, for a chosen cluster (default 5, the starcoder/code cluster):
  expert_attribution/auc_illustration.png   -- 3 panels:
    (A) in-pile vs out-pile routing histograms for the cluster's single best
        expert -> a small, reliable gap => AUC ~ 1 (the signal-to-noise point).
    (B) distribution of one-vs-rest AUC across all 1008 experts -> hundreds are
        individually strong (redundancy, not one signature expert).
    (C) per-expert magnitude |delta| vs separability AUC -> the best separator
        is not the biggest shift (magnitude != separability).
  expert_attribution/auc_illustration.json  -- the exact numbers the report cites.

One-vs-rest AUC is estimated on all in-cluster docs vs a fixed 100k sample of the
rest (rank-based Mann-Whitney; == area under the ROC curve). CPU-only.

    python scripts/modular_extension/auc_illustration.py [--cluster 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1].parent
BASE = ROOT / "modular_extension/cluster/emo100b_step23842"
EMB = BASE / "embeddings_doc_probs.npy"
ASG = BASE / "doc_probs_mean_pca_l2_spherical_kmeans_k64/assignments.npy"
OUT = BASE / "expert_attribution"
N_EXP = 63  # standard experts per layer (layer-major: dim d -> layer d//63, expert d%63)


def per_expert_auc(sub_in, sub_out):
    n_in, n_out = len(sub_in), len(sub_out)
    stacked = np.vstack([sub_in, sub_out])
    auc = np.empty(stacked.shape[1])
    for j in range(stacked.shape[1]):
        r = rankdata(stacked[:, j])
        auc[j] = (r[:n_in].sum() - n_in * (n_in + 1) / 2) / (n_in * n_out)
    return auc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster", type=int, default=5)
    ap.add_argument("--n-out", type=int, default=100_000)
    args = ap.parse_args()
    c = args.cluster

    emb = np.load(EMB, mmap_mode="r")
    asg = np.load(ASG)
    Dd = emb.shape[1]
    mask = asg == c
    idx_in = np.where(mask)[0]
    rng = np.random.default_rng(0)
    idx_out = rng.choice(np.where(~mask)[0], size=args.n_out, replace=False)
    n_in = len(idx_in)

    gmean = emb.mean(0, dtype=np.float64)
    mean_in = emb[idx_in].astype(np.float64).mean(0)
    delta = mean_in - gmean
    absd = np.abs(delta)
    top5_mass = float(np.sort(absd)[::-1][:5].sum() / absd.sum())

    sub_in = np.asarray(emb[idx_in], dtype=np.float32)
    sub_out = np.asarray(emb[idx_out], dtype=np.float32)
    auc = per_expert_auc(sub_in, sub_out)
    sep = np.abs(auc - 0.5) + 0.5           # separability regardless of over/under
    std_in = sub_in.std(0)

    best = int(sep.argmax())                # best separator
    best_mag = int(absd.argmax())           # biggest shift
    strong = np.where(sep >= 0.9)[0]
    d_strong = absd[strong] / np.maximum(std_in[strong], 1e-9)

    stats = {
        "cluster": c,
        "n_in": int(n_in),
        "n_out": int(args.n_out),
        "n_experts": int(Dd),
        "top5_mass": round(top5_mass, 4),
        "counts": {
            "ge_0.95": int((sep >= 0.95).sum()),
            "ge_0.90": int((sep >= 0.90).sum()),
            "ge_0.80": int((sep >= 0.80).sum()),
            "lt_0.55": int((sep < 0.55).sum()),
        },
        "best_expert": {
            "dim": best, "layer": best // N_EXP, "expert": best % N_EXP,
            "auc": round(float(auc[best]), 4),
            "mean_in": round(float(mean_in[best]), 4),
            "mean_global": round(float(gmean[best]), 4),
            "gap_abs_delta": round(float(absd[best]), 4),
            "std_in": round(float(std_in[best]), 4),
            "cohen_d": round(float(absd[best] / max(std_in[best], 1e-9)), 2),
            "mag_rank": int(1 + (absd > absd[best]).sum()),
        },
        "biggest_shift_expert": {
            "dim": best_mag, "gap_abs_delta": round(float(absd[best_mag]), 4),
            "sep_auc": round(float(sep[best_mag]), 3),
            "auc_rank": int(1 + (sep > sep[best_mag]).sum()),
        },
        "strong_experts": {
            "n": int(len(strong)),
            "median_abs_delta": round(float(np.median(absd[strong])), 4),
            "median_std": round(float(np.median(std_in[strong])), 4),
            "median_cohen_d": round(float(np.median(d_strong)), 2),
        },
        "spearman_absdelta_sep": round(float(spearmanr(absd, sep).correlation), 2),
    }
    (OUT / "auc_illustration.json").write_text(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (A) in vs out histograms for the best expert
    a = axes[0]
    lo, hi = 0.0, float(np.percentile(np.concatenate([sub_in[:, best], sub_out[:, best]]), 99.5))
    bins = np.linspace(lo, hi, 60)
    a.hist(sub_out[:, best], bins=bins, density=True, alpha=0.55, color="#94a3b8", label="out-pile (rest)")
    a.hist(sub_in[:, best], bins=bins, density=True, alpha=0.7, color="#2563eb", label="in-pile (cluster)")
    a.set_title(f"(A) one expert (L{best//N_EXP}·E{best%N_EXP}) separates the cluster\n"
                f"gap |δ|={absd[best]:.3f}, spread σ={std_in[best]:.3f}  →  "
                f"d={absd[best]/max(std_in[best],1e-9):.1f}, AUC={auc[best]:.3f}")
    a.set_xlabel("routing probability on this expert"); a.set_ylabel("density")
    a.legend(fontsize=9)

    # (B) distribution of per-expert AUC
    b = axes[1]
    b.hist(sep, bins=np.linspace(0.5, 1.0, 40), color="#7c3aed", alpha=0.85)
    b.axvline(0.9, color="#dc2626", ls="--", lw=1)
    b.set_title(f"(B) most experts are individually strong classifiers\n"
                f"{int((sep>=0.9).sum())}/{Dd} experts reach AUC≥0.9 (redundant, not one)")
    b.set_xlabel("one-vs-rest separability AUC (per expert)"); b.set_ylabel("# experts")

    # (C) magnitude vs separability
    cx = axes[2]
    cx.scatter(absd, sep, s=6, alpha=0.35, color="#475569")
    cx.scatter(absd[best], sep[best], s=70, color="#2563eb", zorder=5,
               label=f"best separator (mag-rank {stats['best_expert']['mag_rank']})")
    cx.scatter(absd[best_mag], sep[best_mag], s=70, color="#dc2626", zorder=5,
               label=f"biggest shift (AUC-rank {stats['biggest_shift_expert']['auc_rank']})")
    cx.set_title(f"(C) magnitude ≠ separability\n"
                 f"Spearman(|δ|, AUC) = {stats['spearman_absdelta_sep']:.2f}")
    cx.set_xlabel("magnitude of mean shift |δ|"); cx.set_ylabel("separability AUC")
    cx.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"How a broad, low-magnitude signature still yields per-expert classifiers "
                 f"(cluster {c}, {n_in:,} docs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "auc_illustration.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "auc_illustration.png", "and auc_illustration.json")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
