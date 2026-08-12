#!/usr/bin/env python3
"""Pre-flight for k=32 CPT: how much do non-expert parameters (attention, embeddings,
norms, router) move during long 64-expert EMO training, relative to expert parameters?

Walks the extend1t run's permanent checkpoints (~100B-token spacing, 100B -> ~835B),
loading only the model weights (no optimizer state) of each, and computes per
parameter group, for each consecutive interval and cumulatively vs the first checkpoint:

  rel_drift = ||theta_t - theta_s||_2 / ||theta_s||_2      (aggregated over the group)
  cosine    = <theta_t, theta_s> / (||theta_t|| ||theta_s||)

Groups: embeddings, attention, norms (block norms), router, experts (fused w1/w2/w3,
all 64 experts incl. the shared one -- the checkpoint stores them fused), lm_head.

This is weight-space drift -- a cheap proxy for functional drift; interpret comparatively
(group vs group), not absolutely.

Outputs (to modular_extension/cluster/emo100b_step23842_100B-130B/k32_cpt/):
  param_drift.json / param_drift.png

Run:  PYTHONPATH=.:src python scripts/modular_extension/param_drift.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from olmo_core.distributed.checkpoint import _load_unsharded_keys
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "models_v2/emo_64exp_50b_wsd_lr2e-3"
OUT = ROOT / "modular_extension/cluster/emo100b_step23842_100B-130B/k32_cpt"
STEPS = [23842, 47684, 71526, 95368, 119210, 143052, 166894, 190736, 199000]
TOK_PER_STEP = 4_194_304


def group_of(key: str) -> str:
    if ".attention." in key:
        return "attention"
    if "_norm." in key:
        return "norms"
    if ".router." in key:
        return "router"
    if ".experts." in key:
        return "experts"
    if "embeddings" in key:
        return "embeddings"
    if "lm_head" in key:
        return "lm_head"
    raise ValueError(key)


def load_model(step: int) -> dict:
    d = str(RUN_DIR / f"step{step}" / "model_and_optim")
    keys = [k for k in RemoteFileSystemReader(d).read_metadata().state_dict_metadata
            if k.startswith("model.")]
    sd = _load_unsharded_keys(d, keys)
    return {k: v.float() for k, v in sd.items()}


def pair_stats(a: dict, b: dict) -> dict:
    """Per-group rel_drift and cosine between two model state dicts."""
    acc: dict = {}
    for k in a:
        g = group_of(k)
        x, y = a[k].flatten(), b[k].flatten()
        s = acc.setdefault(g, {"na": 0.0, "nb": 0.0, "nd": 0.0, "dot": 0.0})
        s["na"] += float(x.pow(2).sum())
        s["nb"] += float(y.pow(2).sum())
        s["nd"] += float((y - x).pow(2).sum())
        s["dot"] += float((x * y).sum())
    return {g: {"rel_drift": (s["nd"] / s["na"]) ** 0.5,
                "cosine": s["dot"] / ((s["na"] ** 0.5) * (s["nb"] ** 0.5))}
            for g, s in acc.items()}


def main():
    OUT.mkdir(exist_ok=True)
    first = prev = load_model(STEPS[0])
    intervals, cumulative = [], []
    for step in STEPS[1:]:
        cur = load_model(step)
        intervals.append({"from_step": STEPS[STEPS.index(step) - 1], "to_step": step,
                          **{"groups": pair_stats(prev, cur)}})
        cumulative.append({"from_step": STEPS[0], "to_step": step,
                           **{"groups": pair_stats(first, cur)}})
        print(f"step {step}: interval {intervals[-1]['groups']}", flush=True)
        if prev is not first:
            del prev
        prev = cur

    out = {"steps": STEPS, "tokens_per_step": TOK_PER_STEP,
           "intervals": intervals, "cumulative": cumulative}
    with open(OUT / "param_drift.json", "w") as f:
        json.dump(out, f, indent=2)

    groups = ["experts", "router", "attention", "embeddings", "norms", "lm_head"]
    colors = {"experts": "#dc2626", "router": "#f59e0b", "attention": "#2563eb",
              "embeddings": "#16a34a", "norms": "#9333ea", "lm_head": "#64748b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, series, title in [
        (axes[0], intervals, "per ~100B-token interval"),
        (axes[1], cumulative, "cumulative since 100B"),
    ]:
        x = [s["to_step"] * TOK_PER_STEP / 1e9 for s in series]
        for g in groups:
            ax.plot(x, [s["groups"][g]["rel_drift"] for s in series], "o-",
                    color=colors[g], label=g)
        ax.set_xlabel("tokens (B)")
        ax.set_title(f"relative L2 drift, {title}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
    axes[0].set_ylabel(r"$\|\Delta\theta\|/\|\theta\|$")
    axes[1].legend(fontsize=9)
    fig.suptitle("Weight drift by parameter group during 64-expert EMO training", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "param_drift.png", dpi=150)
    print(f"Wrote {OUT / 'param_drift.json'} and .png")


if __name__ == "__main__":
    main()
