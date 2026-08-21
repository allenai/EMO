#!/usr/bin/env python3
# PARENT: "scripts/modular_extension/aggregate_k32cpt_evals.py"
"""Aggregate meta_learning phase-2 k=32 CPT eval outputs into one results JSON + report
figures (full-128-expert view).

Reads the evals dir (per-checkpoint shard JSONs from eval_k32cpt_ce.py):
  ce_<arm>_base_20B_shard*.json           the arm's step4768 pre-CPT anchor
  ce_vanilla_upper_40B_shard*.json        vanilla step9537 (plain continued pretraining)
                                          on VANILLA's clusters
  ce_sametok_clusters_vanilla40B_shard*.json   same 40B model on SAMETOK's clusters
  ce_<arm>_after_cXX_shard*.json          per-stage pool snapshots

Only checkpoints with all 8 shards are included. All deltas are vs the arm's own 20B
base anchor (identical 25M-token eval prefix per cluster).

Outputs (to <evals>/ and mirrored to claude_outputs/meta_learning/):
  k32cpt_results.json
  k32cpt_heatmap_<arm>.png     stage x evaluated-cluster CE delta vs 20B base
  k32cpt_curves.png            just-trained / mean / forgetting summary per arm

Run:  python scripts/meta_learning/aggregate_k32cpt_evals.py
"""
import glob
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/root/EMO/meta_learning/k32_cpt_runs")
EV = RUNS / "evals"
OUT_MIRROR = Path("/root/EMO/claude_outputs/meta_learning")
ARMS = ["sametok_ws_lam05", "vanilla"]
BASE_TAG = {"sametok_ws_lam05": "sametok_ws_lam05_base_20B", "vanilla": "vanilla_base_20B"}
UPPER_TAG = {"sametok_ws_lam05": "sametok_clusters_vanilla40B", "vanilla": "vanilla_upper_40B"}
COLORS = {"sametok_ws_lam05": "#7c3aed", "vanilla": "#2563eb"}
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
    try:
        for line in open(RUNS / arm / "order.txt"):
            vals = [int(x) for x in line.split()]
            if len(vals) == K:
                return vals
    except FileNotFoundError:
        pass
    return list(range(K))


def main():
    res = {"anchors": {}, "arms": {}}
    for arm in ARMS:
        base = load_tag(BASE_TAG[arm])
        upper = load_tag(UPPER_TAG[arm])
        assert base is not None, f"missing base anchor for {arm}"
        res["anchors"][arm] = {"base_20B": base, "upper_40B": upper}
        order = arm_order(arm)
        stages = []
        for pos, c in enumerate(order):
            ce = load_tag(f"{arm}_after_c{c:02d}")
            if ce:
                stages.append({"stage": pos, "trained_cluster": c, "ce": ce})
        res["arms"][arm] = {"order": order, "stages": stages}

    with open(EV / "k32cpt_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---- heatmaps: delta vs the arm's 20B base, columns in training order (diagonal =
    # just-trained); final separated row = the 40B plain-CPT upper bound's delta.
    for arm in ARMS:
        st = res["arms"][arm]["stages"]
        if not st:
            continue
        order = res["arms"][arm]["order"]
        base = res["anchors"][arm]["base_20B"]
        upper = res["anchors"][arm]["upper_40B"]
        n_rows = len(st) + (1 if upper else 0)
        M = np.full((n_rows, K), np.nan)
        for i, s in enumerate(st):
            for j, c in enumerate(order):
                M[i, j] = s["ce"][c] - base[c]
        if upper:
            for j, c in enumerate(order):
                M[-1, j] = upper[c] - base[c]
        fig, ax = plt.subplots(figsize=(10.5, max(3.4, 0.3 * n_rows + 1.6)))
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        for i, s in enumerate(st):
            j = order.index(s["trained_cluster"])
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=1.4, ec="black"))
        labels = [f"{s['stage']}: c{s['trained_cluster']}" for s in st]
        if upper:
            ax.axhline(len(st) - 0.5, color="black", lw=1.6)
            labels.append("40B plain CPT")
        ax.set_xticks(range(K), [str(c) for c in order], fontsize=7)
        ax.set_yticks(range(n_rows), labels, fontsize=7)
        ax.set_xlabel("evaluated cluster (in training order)")
        ax.set_ylabel("after stage (trained cluster)")
        ax.set_title(f"{arm}: full-128e held-out CE delta vs 20B start (black box = "
                     "just-trained; bottom row = vanilla 40B plain CPT)", fontsize=11)
        fig.colorbar(im, ax=ax, label="CE delta (nats; negative = better)")
        fig.tight_layout()
        fig.savefig(EV / f"k32cpt_heatmap_{arm}.png", dpi=150)
        plt.close(fig)

    # ---- curves: just-trained delta, mean delta, prior-trained mean (forgetting)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True)
    for arm in ARMS:
        st = res["arms"][arm]["stages"]
        if not st:
            continue
        base = res["anchors"][arm]["base_20B"]
        upper = res["anchors"][arm]["upper_40B"]
        x = [s["stage"] for s in st]
        just = [s["ce"][s["trained_cluster"]] - base[s["trained_cluster"]] for s in st]
        mean_all = [np.mean([s["ce"][c] - base[c] for c in range(K)]) for s in st]
        prior = []
        for s in st:
            before = [t["trained_cluster"] for t in st if t["stage"] < s["stage"]]
            prior.append(np.mean([s["ce"][c] - base[c] for c in before]) if before else np.nan)
        c = COLORS[arm]
        axes[0].plot(x, just, "o-", color=c, label=arm)
        axes[1].plot(x, mean_all, "o-", color=c, label=arm)
        axes[2].plot(x, prior, "o-", color=c, label=arm)
        if upper:
            um = np.mean([upper[k] - base[k] for k in range(K)])
            for ax in axes:
                ax.axhline(um, color=c, ls="--", lw=1.2, alpha=0.7)
    for ax, title in [(axes[0], "just-trained cluster (specialization)"),
                      (axes[1], "mean over all 32 clusters"),
                      (axes[2], "mean over previously-trained clusters (forgetting)")]:
        ax.axhline(0, color="#94a3b8", lw=1)
        ax.set_xlabel("stage")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out CE delta vs 20B base (nats)")
    axes[0].legend(fontsize=9)
    fig.suptitle("phase-2 k=32 CPT: full-model held-out CE deltas vs each arm's 20B base "
                 "(dashed = vanilla 40B plain-CPT reference on that arm's clusters)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(EV / "k32cpt_curves.png", dpi=150)
    plt.close(fig)

    OUT_MIRROR.mkdir(parents=True, exist_ok=True)
    for name in ["k32cpt_heatmap_sametok_ws_lam05.png", "k32cpt_heatmap_vanilla.png",
                 "k32cpt_curves.png", "k32cpt_results.json"]:
        if (EV / name).exists():
            shutil.copy(EV / name, OUT_MIRROR / name)
    done = {arm: len(res["arms"][arm]["stages"]) for arm in ARMS}
    print(json.dumps({"stages_evaluated": done,
                      "mirrored_to": str(OUT_MIRROR)}))


if __name__ == "__main__":
    main()
