#!/usr/bin/env python3
"""Pre-flight for k=32 CPT: how much router probability mass do the top-32 experts
(selected per layer from each cluster's mean routing) capture, vs the same selection
on random document sets?

For each of the 32 clusters (frozen partition global_k32_seed1): compute the mean
doc_probs vector over the cluster's docs (16 layers x 63 standard experts, each layer's
softmax sums to 1), sort experts per layer, and record the mass captured by the top-32
(the trainable working set) vs the bottom-31. Baseline: one PERMUTATION of the cluster
labels -> 32 random doc sets with exactly the same sizes, same statistic. The gap is
what clustering buys in expert-usage concentration.

Outputs (to <data-dir>/k32_cpt/):
  expert_concentration.json   per-cluster & per-random-set top-32 mass (mean over layers,
                              plus per-layer detail and the selected expert ids per layer)
  expert_concentration.png    sorted bars vs the random band

Run:  PYTHONPATH=.:src python scripts/modular_extension/cluster_expert_concentration.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B"
OUT = BASE / "k32_cpt"
K = 32
N_LAYERS, N_EXP = 16, 63
TOP = 32
PERM_SEED = 7


def cluster_means(pre_path, labels, k, chunk=2_000_000):
    """(k, D) mean embedding per label via one chunked pass."""
    emb = np.load(pre_path, mmap_mode="r")
    n, d = emb.shape
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        block = np.asarray(emb[lo:hi], dtype=np.float32)
        S = sparse.csr_matrix(
            (np.ones(hi - lo, np.float32), labels[lo:hi], np.arange(hi - lo + 1)),
            shape=(hi - lo, k),
        )
        sums += (S.T @ block).astype(np.float64)
    return sums / counts[:, None], counts


def top_mass(mean_vec):
    """Per-layer top-TOP mass of a (1008,) mean-prob vector; returns (per_layer, ids)."""
    m = mean_vec.reshape(N_LAYERS, N_EXP)
    m = m / m.sum(axis=1, keepdims=True)  # renormalize (fp16 accumulation noise)
    order = np.argsort(-m, axis=1)
    per_layer = np.array([m[i, order[i, :TOP]].sum() for i in range(N_LAYERS)])
    return per_layer, order[:, :TOP]


def main():
    OUT.mkdir(exist_ok=True)
    labels = np.load(BASE / "k_selection/global_k32_seed1/labels.npy").astype(np.int32)
    rng = np.random.default_rng(PERM_SEED)
    perm_labels = labels[rng.permutation(labels.shape[0])]

    means, counts = cluster_means(BASE / "embeddings_doc_probs.npy", labels, K)
    rmeans, _ = cluster_means(BASE / "embeddings_doc_probs.npy", perm_labels, K)
    gmean = (means * counts[:, None]).sum(axis=0) / counts.sum()

    res = {"clusters": [], "random_sets": [], "global": {}}
    for c in range(K):
        per_layer, ids = top_mass(means[c])
        res["clusters"].append({
            "cluster": c,
            "num_docs": int(counts[c]),
            "top32_mass": round(float(per_layer.mean()), 4),
            "top32_mass_per_layer": [round(float(x), 4) for x in per_layer],
            "top32_experts_per_layer": ids.tolist(),
        })
        per_layer_r, _ = top_mass(rmeans[c])
        res["random_sets"].append({
            "set": c,
            "num_docs": int(counts[c]),
            "top32_mass": round(float(per_layer_r.mean()), 4),
        })
    gl, _ = top_mass(gmean)
    res["global"] = {"top32_mass": round(float(gl.mean()), 4)}

    with open(OUT / "expert_concentration.json", "w") as f:
        json.dump(res, f, indent=2)

    cl = sorted((c["top32_mass"] for c in res["clusters"]), reverse=True)
    rnd = [r["top32_mass"] for r in res["random_sets"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(K), cl, color="#2563eb", label="cluster docs (top-32 chosen per cluster)")
    ax.axhspan(min(rnd), max(rnd), color="#d97706", alpha=0.25,
               label=f"random doc sets, same sizes (range over {K} sets)")
    ax.axhline(res["global"]["top32_mass"], color="#d97706", ls="--", lw=1.2,
               label="all docs (global top-32)")
    ax.set_xlabel("cluster (sorted by concentration)")
    ax.set_ylabel("router mass in top-32 experts")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Probability mass captured by each cluster's 32-expert working set "
                 "(mean over 16 layers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "expert_concentration.png", dpi=150)
    print(json.dumps({"clusters_minmax": (cl[-1], cl[0]),
                      "random_minmax": (min(rnd), max(rnd)),
                      "global": res["global"]["top32_mass"]}))


if __name__ == "__main__":
    main()
