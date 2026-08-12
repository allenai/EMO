#!/usr/bin/env python3
"""Plot subsample-stability results: agreement with the full-fit oracle partition
vs subsample size, honest vs frozen-transform arms, with the full-data seed-noise
ceiling as a band. Reads scores.json produced by subsample_stability.py score.

Run:  python scripts/modular_extension/subsample_stability_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B/subsample_stability"


def main():
    scores = json.load(open(SUB / "scores.json"))
    n_total = next(iter(scores.values()))["n_rows_assigned"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in [
        (axes[0], "acc_tokens", "token-weighted matched accuracy"),
        (axes[1], "ari", "adjusted Rand index"),
    ]:
        for mode, color in [("honest", "#d97706"), ("frozen", "#2563eb")]:
            arms = [s for s in scores.values() if s["mode"] == mode]
            by_n: dict = {}
            for s in arms:
                by_n.setdefault(s["n_sample"], []).append(s[metric])
            ns = sorted(by_n)
            means = [sum(by_n[n]) / len(by_n[n]) for n in ns]
            ax.plot(ns, means, "o-", color=color, label=f"{mode} fit", zorder=3)
            for n in ns:  # individual seeds where we have >1 draw
                for v in by_n[n]:
                    ax.plot([n], [v], "o", color=color, mfc="none", ms=9, zorder=2)
        ceil = [s[metric] for s in scores.values() if s["mode"] == "fullseed"]
        if ceil:
            ax.axhspan(min(ceil), max(ceil), color="#16a34a", alpha=0.15, zorder=1)
            ax.axhline(sum(ceil) / len(ceil), color="#16a34a", ls="--", lw=1,
                       label=f"full-data re-fit, new seed (n={n_total/1e6:.1f}M)")
        ax.set_xscale("log")
        ax.set_xlabel("documents the clustering was fit on")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("agreement with full-fit partition")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("How much of the full-data partition does a subsample fit recover?",
                 fontsize=12)
    fig.tight_layout()
    out = SUB / "substab_agreement.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
