#!/usr/bin/env python3
"""Plot k-selection results: criterion curves (objective elbow, silhouette,
Davies-Bouldin) for the global fit and the three 1M-doc draws, and stability vs k
(global seed pairs, subsample cross-draw, subsample->global recovery).
Reads k_selection_summary.json produced by k_selection.py aggregate.

Run:  python scripts/modular_extension/k_selection_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
KSEL = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B/k_selection"


def characterize_k4():
    """Sizes + top sources of the four stable k=4 clusters (global_k4_seed1),
    saved for the report."""
    import collections
    import re

    import numpy as np

    lab = np.load(KSEL / "global_k4_seed1/labels.npy")
    ids = np.load(KSEL.parent / "doc_ids.npz", allow_pickle=True)
    src_idx, dl = ids["source_index"], ids["doc_len"]
    short = [re.search(r"preprocessed/([^/]+(?:/[^/]+)?)/", str(p)).group(1)
             if re.search(r"preprocessed/", str(p)) else str(p)
             for p in ids["source_paths"]]
    out = []
    for c in range(4):
        m = lab == c
        cnt: collections.Counter = collections.Counter()
        for si, n in zip(*np.unique(src_idx[m], return_counts=True)):
            cnt[short[si]] += int(n)
        out.append({
            "cluster": c,
            "num_docs": int(m.sum()),
            "num_tokens": int(dl[m].sum()),
            "top_sources": {s: round(n / m.sum(), 4) for s, n in cnt.most_common(3)},
        })
    with open(KSEL / "k4_characterization.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {KSEL / 'k4_characterization.json'}")


def main():
    summ = json.load(open(KSEL / "k_selection_summary.json"))
    curves, stab = summ["curves"], summ["stability"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # -- criterion curves ---------------------------------------------------
    styles = {"global": dict(color="#16a34a", lw=2.2, zorder=3),
              "sub1M": dict(color="#94a3b8", lw=1.2, zorder=2)}
    for ax, metric, title in [
        (axes[0], "objective", "spherical k-means objective\n(mean cosine to own centroid)"),
        (axes[1], "silhouette_cosine", "silhouette (cosine, 50K sample)"),
    ]:
        for key, cur in sorted(curves.items()):
            kind = "global" if key.startswith("global") else "sub1M"
            label = None
            if key.endswith(("seed0", "seed1")) or kind == "global":
                label = {"global": "full 27.5M fit", "sub1M": "1M-doc draws"}[kind]
                if label in [ln.get_label() for ln in ax.get_lines()]:
                    label = None
            ax.plot(cur["k"], cur[metric], "o-", label=label, **styles[kind])
        ax.set_xscale("log", base=2)
        ax.set_xticks(curves[next(iter(curves))]["k"],
                      [str(k) for k in curves[next(iter(curves))]["k"]])
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # -- stability vs k -----------------------------------------------------
    ax = axes[2]
    for name, series, color in [
        ("full-data fits, two seeds", stab["global"], "#16a34a"),
        ("two 1M draws, same k", stab["sub1M"], "#2563eb"),
        ("1M draw vs full-data fit", stab["recover"], "#d97706"),
    ]:
        ks = sorted(int(k) for k in series)
        ax.plot(ks, [series[str(k)]["ari"] if str(k) in series else series[k]["ari"]
                     for k in ks], "o-", color=color, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("ARI between the two partitions")
    ax.set_title("reproducibility vs k", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Is there a principled k, and does it fix the seed variance?", fontsize=12)
    fig.tight_layout()
    out = KSEL / "ksel_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    characterize_k4()


if __name__ == "__main__":
    main()
