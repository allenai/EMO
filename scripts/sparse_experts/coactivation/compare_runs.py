#!/usr/bin/env python3
"""
Cross-model comparison figures for the co-activation analysis (e.g. 8-of-512 vs 8-of-1024).

Reads the analysis.json written by analyze_coactivation.py for each run and draws per-layer
overlays: modularity (spectral + Louvain), effective experts (absolute and as a fraction of the
standard experts), token-vs-document lift agreement, pair statistics (never-co-active share,
lift median / p99, conditional p99.9), per-source modularity, and experts touched per document.

  python scripts/sparse_experts/coactivation/compare_runs.py \
      --run 512=claude_outputs/sparse_experts/coactivation/sparse_8of512_10b_step2384/analysis.json \
      --run 1024=claude_outputs/sparse_experts/coactivation/sparse_8of1024_10b_step2384/analysis.json \
      --out claude_outputs/sparse_experts/coactivation/compare
"""

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"512": "#1f77b4", "1024": "#d62728"}
STYLES = {"pool_config": "-", "pool_64": "--"}


def save_png(fig, path, dpi):
    fig.savefig(path, dpi=dpi)
    try:
        from PIL import Image

        Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
            path, optimize=True
        )
    except ImportError:
        pass


def series(pool, key_fn):
    L = len(pool["layers"])
    return [key_fn(pool["layers"][str(li)]["sample"]) for li in range(L)]


def n_std_experts(pool):
    # independence baseline is 7 / E_s -> recover E_s from the stored baseline
    c = pool["layers"]["0"]["sample"].get("conditional")
    return int(round(7.0 / c["independence_baseline"])) if c else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="label=path/to/analysis.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = json.load(open(path))["pools"]
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(label):
        return COLORS.get(label, None)

    # 1) summary: Q, effective experts (abs + fraction), rho -- both pools
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for pool_key, row in [("pool_config", 0), ("pool_64", 1)]:
        for label, pools in runs.items():
            pool = pools.get(pool_key)
            if pool is None:
                continue
            Es = n_std_experts(pool)
            q = series(pool, lambda s: s["structure"]["Q_spectral"])
            ql = series(pool, lambda s: s["structure"]["Q_louvain"])
            eff = series(pool, lambda s: s["usage"]["effective_experts"])
            rho = series(pool, lambda s: s["token_vs_doc_spearman"])
            axes[row, 0].plot(q, marker="o", color=col(label), label=f"{label} spectral k=8")
            axes[row, 0].plot(
                ql, marker="s", ls="--", color=col(label), alpha=0.6, label=f"{label} Louvain"
            )
            axes[row, 1].plot(
                np.array(eff) / Es, marker="o", color=col(label), label=f"{label} (of {Es})"
            )
            axes[row, 2].plot(rho, marker="o", color=col(label), label=label)
        title = "full routing" if pool_key == "pool_config" else "pool pinned to 64"
        axes[row, 0].set_title(f"{title}: modularity Q")
        axes[row, 1].set_title(f"{title}: effective experts / #standard experts")
        axes[row, 2].set_title(f"{title}: Spearman(token lift, doc lift)")
        axes[row, 1].set_ylim(0, 1)
        for ax in axes[row]:
            ax.set_xlabel("layer")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, args.out / "compare_summary.png", 100)
    plt.close(fig)

    # 2) pair statistics (full routing)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for label, pools in runs.items():
        pool = pools["pool_config"]
        axes[0].plot(
            series(pool, lambda s: 100 * s["pairs"]["never_coactive_frac"]),
            marker="o",
            color=col(label),
            label=label,
        )
        axes[1].plot(
            series(pool, lambda s: s["pairs"]["lift_median"]),
            marker="o",
            color=col(label),
            label=label,
        )
        axes[2].plot(
            series(pool, lambda s: s["pairs"]["lift_p99"]),
            marker="o",
            color=col(label),
            label=label,
        )
        axes[3].plot(
            series(pool, lambda s: s["conditional"]["p999"]),
            marker="o",
            color=col(label),
            label=label,
        )
    axes[0].set_title("% pairs never co-active")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].set_title("lift median")
    axes[2].set_title("lift p99")
    axes[3].set_title("conditional P(j|i) p99.9")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, args.out / "compare_pairs.png", 100)
    plt.close(fig)

    # 3) per-source Q (full routing), one panel per model
    labels = list(runs)
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        pool = runs[label]["pool_config"]
        L = len(pool["layers"])
        for s, v in pool["per_source"].items():
            ax.plot([v["layers"][str(li)]["Q_spectral"] for li in range(L)], marker="o", label=s)
        ax.plot(
            series(pool, lambda s: s["structure"]["Q_spectral"]),
            color="k",
            lw=2,
            label="pooled (all sources)",
        )
        ax.set_title(f"{label} experts: modularity Q per source")
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, args.out / "compare_per_source.png", 100)
    plt.close(fig)

    # 4) experts per doc (abs + fraction)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for label, pools in runs.items():
        for pool_key, ls in STYLES.items():
            pool = pools.get(pool_key)
            if pool is None:
                continue
            Es = n_std_experts(pool)
            u = pool["unique_experts_per_doc"]["mean_by_layer"]
            nm = f"{label} {'full' if pool_key == 'pool_config' else 'pool 64'}"
            axes[0].plot(u, marker="o", ls=ls, color=col(label), label=nm)
            axes[1].plot(np.array(u) / Es, marker="o", ls=ls, color=col(label), label=nm)
    axes[0].set_title("distinct routed experts per document (mean)")
    axes[1].set_title("as a fraction of the standard experts")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, args.out / "compare_unique_per_doc.png", 100)
    plt.close(fig)

    # digest json
    digest = {}
    for label, pools in runs.items():
        d = {}
        for pool_key, pool in pools.items():
            Es = n_std_experts(pool)
            L = len(pool["layers"])
            q = series(pool, lambda s: s["structure"]["Q_spectral"])
            eff = series(pool, lambda s: s["usage"]["effective_experts"])
            d[pool_key] = dict(
                num_std_experts=Es,
                Q_mean_layers_1plus=float(np.mean(q[1:])),
                Q_layer0=q[0],
                eff_experts_mean=float(np.mean(eff)),
                eff_fraction_mean=float(np.mean(eff) / Es) if Es else None,
                rho_mean=float(np.mean(series(pool, lambda s: s["token_vs_doc_spearman"]))),
                never_mean_layers_3plus=float(
                    np.mean(series(pool, lambda s: s["pairs"]["never_coactive_frac"])[3:])
                ),
                lift_p99_mean=float(np.mean(series(pool, lambda s: s["pairs"]["lift_p99"]))),
                experts_per_doc_mean=float(
                    np.mean(pool["unique_experts_per_doc"]["mean_by_layer"])
                ),
                per_source_Q_mean={
                    s: float(np.mean([v["layers"][str(li)]["Q_spectral"] for li in range(2, L)]))
                    for s, v in pool["per_source"].items()
                },
            )
        digest[label] = d
    json.dump(digest, open(args.out / "digest.json", "w"), indent=1)
    print(json.dumps(digest, indent=1))


if __name__ == "__main__":
    main()
