#!/usr/bin/env python3
"""Aggregate k=32 CPT eval outputs into one results JSON + report figures.

Reads the evals dir (per-checkpoint shard JSONs from eval_k32cpt_ce.py):
  ce_step23842_100B[_25M]_shard*.json     100B start anchor (50M / 25M tokens per cluster)
  ce_baseline_130B[_25M]_shard*.json      130B normally-trained baseline
  ce_<arm>_after_cXX_shard*.json          per-stage pool snapshots (25M tokens per cluster)

Only checkpoints with all 8 shards are included. Snapshot deltas are computed against the
25M anchors when present (identical eval prefix), else the 50M anchors (flagged in JSON).

Outputs (to <evals>/):
  k32cpt_results.json
  k32cpt_heatmap_<arm>.png     stage x evaluated-cluster CE delta vs 100B anchor
  k32cpt_curves.png            just-trained delta, mean delta, forgetting summary per arm

Run:  python scripts/modular_extension/aggregate_k32cpt_evals.py
"""
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/root/EMO/modular_extension/k32_cpt_runs")
EV = RUNS / "evals"
ARMS = ["carry", "carry_shuf", "fresh"]
K = 32


def load_tag(tag):
    shards = glob.glob(str(EV / f"ce_{tag}_shard*.json"))
    if len(shards) != 8:
        return None
    out = {}
    for p in shards:
        out.update({int(c): v["ce"] for c, v in json.load(open(p))["results"].items()})
    return out if len(out) == K else None


def arm_order(arm):
    # order.txt gains a line per driver invocation; pilot lines list a subset.
    for line in open(RUNS / arm / "order.txt"):
        vals = [int(x) for x in line.split()]
        if len(vals) == K:
            return vals
    return list(range(K))


def main():
    res = {"anchors": {}, "arms": {}}
    for tag in ("step23842_100B", "baseline_130B", "step23842_100B_25M", "baseline_130B_25M"):
        ce = load_tag(tag)
        if ce:
            res["anchors"][tag] = ce
    use25 = "step23842_100B_25M" in res["anchors"] and "baseline_130B_25M" in res["anchors"]
    a100 = res["anchors"]["step23842_100B_25M" if use25 else "step23842_100B"]
    abase = res["anchors"]["baseline_130B_25M" if use25 else "baseline_130B"]
    res["delta_reference"] = "25M anchors" if use25 else "50M anchors (25M snapshots -- small prefix mismatch)"

    for arm in ARMS:
        order = arm_order(arm)
        stages = []
        for pos, c in enumerate(order):
            ce = load_tag(f"{arm}_after_c{c:02d}")
            if ce:
                stages.append({"stage": pos, "trained_cluster": c, "ce": ce})
        res["arms"][arm] = {"order": order, "stages": stages}

    with open(EV / "k32cpt_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---- heatmaps: delta vs 100B, columns in the arm's TRAINING order (diagonal = trained)
    for arm in ARMS:
        st = res["arms"][arm]["stages"]
        if not st:
            continue
        order = res["arms"][arm]["order"]
        M = np.full((len(st), K), np.nan)
        for i, s in enumerate(st):
            for j, c in enumerate(order):
                M[i, j] = s["ce"][c] - a100[c]
        fig, ax = plt.subplots(figsize=(10.5, max(3.2, 0.3 * len(st) + 1.6)))
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        for i, s in enumerate(st):
            j = order.index(s["trained_cluster"])
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=1.4, ec="black"))
        ax.set_xticks(range(K), [str(c) for c in order], fontsize=7)
        ax.set_yticks(range(len(st)),
                      [f"{s['stage']}: c{s['trained_cluster']}" for s in st], fontsize=7)
        ax.set_xlabel("evaluated cluster (in training order)")
        ax.set_ylabel("after stage (trained cluster)")
        ax.set_title(f"{arm}: held-out CE delta vs 100B start (black box = just-trained)",
                     fontsize=11)
        fig.colorbar(im, ax=ax, label="CE delta (nats; negative = better)")
        fig.tight_layout()
        fig.savefig(EV / f"k32cpt_heatmap_{arm}.png", dpi=150)
        plt.close(fig)

    # ---- curves: just-trained delta, mean delta, prior-trained mean (forgetting)
    base_mean = np.mean([abase[c] - a100[c] for c in range(K)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True)
    colors = {"carry": "#2563eb", "carry_shuf": "#16a34a", "fresh": "#d97706"}
    for arm in ARMS:
        st = res["arms"][arm]["stages"]
        if not st:
            continue
        x = [s["stage"] for s in st]
        just = [s["ce"][s["trained_cluster"]] - a100[s["trained_cluster"]] for s in st]
        mean_all = [np.mean([s["ce"][c] - a100[c] for c in range(K)]) for s in st]
        prior = []
        for s in st:
            trained_before = [t["trained_cluster"] for t in st if t["stage"] < s["stage"]]
            prior.append(np.mean([s["ce"][c] - a100[c] for c in trained_before])
                         if trained_before else np.nan)
        axes[0].plot(x, just, "o-", color=colors[arm], label=arm)
        axes[1].plot(x, mean_all, "o-", color=colors[arm], label=arm)
        axes[2].plot(x, prior, "o-", color=colors[arm], label=arm)
    for ax, title in [(axes[0], "just-trained cluster (specialization)"),
                      (axes[1], "mean over all 32 clusters"),
                      (axes[2], "mean over previously-trained clusters (forgetting)")]:
        ax.axhline(0, color="#94a3b8", lw=1)
        ax.axhline(base_mean, color="#dc2626", ls="--", lw=1.2,
                   label="130B baseline (mean)" if ax is axes[0] else None)
        ax.set_xlabel("stage")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out CE delta vs 100B (nats)")
    axes[0].legend(fontsize=9)
    fig.suptitle("k=32 CPT: per-stage held-out CE deltas vs the 100B start "
                 f"(reference: {res['delta_reference']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(EV / "k32cpt_curves.png", dpi=150)

    n_st = {a: len(res["arms"][a]["stages"]) for a in ARMS}
    print(json.dumps({"stages_evaluated": n_st, "delta_reference": res["delta_reference"],
                      "baseline_mean_delta": round(float(base_mean), 4)}))


if __name__ == "__main__":
    main()
