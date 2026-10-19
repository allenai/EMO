#!/usr/bin/env python3
"""Merged four-square sub-models (sub-model LR 4e-4) vs joint training, EMO 512e / 1000e / 2000e, 10B -> 61B.

Held-out cross-entropy on the 300B-stream sample (7,993 instances, 65.5M tokens; sparse_experts/olmoe3_routing/sample_8k_300b.npz),
read from the merged routing passes of the olmoe3_routing experiment (runs_heldout300b_*/.../none/meta.json). The raw values are
embedded below so the figure can be regenerated anywhere:   python plot_merge_vs_joint.py  -> merge_vs_joint.png / .pdf
  * baseline       the start model continued jointly on the whole stream (constant LR 8e-4), matched checkpoints
  * merged squares the same 10B start model split into 4 random sub-models (a quarter of the experts in every layer 2-9, layer 1 whole),
                   each trained at LR 4e-4 on a quarter of the documents (random packs to 30B, contiguous stream quarters 30B -> 61B),
                   experts put side by side and shared parameters averaged
  * start model    the 10B checkpoint both lines start from (dotted)
Points: 10B (step 19,074), 15.7B (30,000), 20B (38,148), 30B (57,221), 61B (116,479); 524,288 tokens per step.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [19074, 30000, 38148, 57221, 116479]
TOKENS_B = [s * 524288 / 1e9 for s in STEPS]           # 10.0, 15.7, 20.0, 30.0, 61.1
LABELS = ["10B", "15.7B", "20B", "30B", "61B"]
DATA = {  # held-out CE (300B sample); first value = the shared 10B start model
    "EMO 512e":  dict(baseline=[2.4654, 2.4204, 2.4031, 2.3758, 2.3310], merged=[2.4654, 2.4168, 2.4032, 2.3849, 2.3514]),
    "EMO 1000e": dict(baseline=[2.4440, 2.3975, 2.3788, 2.3503, 2.3128], merged=[2.4440, 2.3908, 2.3727, 2.3513, 2.3149]),
    "EMO 2000e": dict(baseline=[2.4288, 2.3757, 2.3620, 2.3359, 2.2975], merged=[2.4288, 2.3673, 2.3497, 2.3241, 2.2797]),
}
C_BASE, C_MERGE, C_START, INK, MUTED = "#2a78d6", "#eb6834", "#6b6a66", "#0b0b0b", "#52514e"


def main(out="merge_vs_joint"):
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#9a9893", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, (name, d) in zip(axes, DATA.items()):
        ax.plot(TOKENS_B, d["baseline"], color=C_BASE, lw=2, marker="o", ms=5.5, label="baseline (joint training)")
        ax.plot(TOKENS_B, d["merged"], color=C_MERGE, lw=2, marker="s", ms=5.5, label="merged 4 squares, sub-model LR 4e-4")
        ax.axhline(d["baseline"][0], color=C_START, lw=1.4, ls=":", label="start model (10B)")
        yb, ym = d["baseline"][-1], d["merged"][-1]; close = abs(ym - yb) < 0.006   # direct labels at the right end; pushed apart when the lines meet
        ax.annotate(f"{yb:.3f}", (TOKENS_B[-1], yb), xytext=(6, -6 if close else 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
        ax.annotate(f"{ym:.3f}", (TOKENS_B[-1], ym), xytext=(6, 6 if close else 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
        gap = ym - yb
        ax.set_title(f"{name}  (merged {'+' if gap > 0 else '-'}{abs(gap):.3f} vs joint at 61B)", fontsize=11.5, color=INK)
        ax.set_xscale("log"); ax.set_xticks(TOKENS_B, LABELS); ax.set_xticks([], minor=True); ax.set_xlim(9, 75); ax.set_xlabel("tokens trained")
        ax.grid(axis="y", color="#e8e7e3", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("held-out CE (300B-stream sample)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Four random squares at sub-model LR 4e-4 vs joint training, from the same 10B EMO start model", fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"): fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out + ".png", out + ".pdf")


if __name__ == "__main__":
    main()
