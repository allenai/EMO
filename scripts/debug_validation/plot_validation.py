"""Figures for the debug_validation report: v3-small ppl validation CE per set and the pretraining
CE loss, for the four OLMoE3-ladder 275M 512-expert arms (standard router; EMO with uniform,
Beta(2,1) and Beta(4,1) per-document pool sizes).

Inputs (claude_outputs/debug_validation/ppl_validation/): summary.json (offline eval of the
standard + uniform-EMO checkpoints, 5 each), inloop_beta_arms.json (the Beta arms' in-loop evals
every 1000 steps), train_ce_curves.json (W&B train/CE loss of all four runs).
Outputs: claude_outputs/debug_validation/figs/{val_<set>.png, val_mean.png, val_delta.png,
train_ce.png, train_ce_zoom.png, train_ce_delta.png}.

    python scripts/debug_validation/plot_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "claude_outputs/debug_validation/ppl_validation"
OUT = ROOT / "claude_outputs/debug_validation/figs"
SETS = [
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
]
TOKENS_PER_STEP = 524_288

# dataviz reference palette (light): categorical slots 1-3 validated all-pairs; the standard
# router is the reference line and wears the secondary text tone, not a series hue.
ARMS = {
    "uniform": dict(label="EMO, uniform pool [16, 512]", color="#2a78d6", ls="-"),
    "beta2": dict(label="EMO, Beta(2,1) pool", color="#eb6834", ls="-"),
    "beta4": dict(label="EMO, Beta(4,1) pool", color="#1baf7a", ls="-"),
    "standard": dict(label="standard router (reference)", color="#52514e", ls="--"),
}
SURFACE, GRID, TEXT, MUTED = "#fcfcfb", "#e6e5e1", "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def load():
    summary = json.load(open(IN / "summary.json"))
    inloop = json.load(open(IN / "inloop_beta_arms.json"))
    series = {}  # arm -> set -> sorted list of (step, ce)
    for arm, run in (("standard", "olmoe3_275m_10b"), ("uniform", "olmoe3_275m_emo_10b")):
        series[arm] = {
            s: sorted(
                (int(st), d["per_set"][f"{s}-validation"]["CE loss"])
                for st, d in summary[run].items()
            )
            for s in SETS
        }
    for arm, run in (
        ("beta2", "olmoe3_275m_emo_beta2_10b"),
        ("beta4", "olmoe3_275m_emo_beta4_10b"),
    ):
        series[arm] = {
            s: sorted((int(st), d[s]) for st, d in inloop[run].items() if s in d) for s in SETS
        }
    for arm in series:
        pts = {}
        for s in SETS:
            for st, ce in series[arm][s]:
                pts.setdefault(st, []).append(ce)
        series[arm]["mean"] = sorted(
            (st, float(np.mean(v))) for st, v in pts.items() if len(v) == len(SETS)
        )
    return series


def style_axes(ax, ylabel="CE loss (nats/token)"):
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def legend_below(ax, ncol=2):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol, handlelength=2.2)


def end_labels(ax, finals, fmt="{:.3f}", min_gap_frac=0.07):
    """finals: list of (x, y, color). Labels at the right edge, pushed apart vertically so they
    never collide, joined to their line ends by thin leader lines. Text wears the text tone."""
    if not finals:
        return
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * min_gap_frac
    order = sorted(range(len(finals)), key=lambda i: finals[i][1])
    ys = [finals[i][1] for i in order]
    for k in range(1, len(ys)):  # push up
        ys[k] = max(ys[k], ys[k - 1] + gap)
    over = ys[-1] - (y1 - gap * 0.5)
    if over > 0:  # slide the stack down if it ran off the top
        ys = [y - over for y in ys]
    xmax = max(x for x, _, _ in finals)
    x_txt = xmax + 0.55
    for k, i in enumerate(order):
        x, y, c = finals[i]
        ax.plot([x + 0.08, x_txt - 0.05], [y, ys[k]], color=c, lw=0.8, alpha=0.7, zorder=2)
        ax.text(x_txt, ys[k], fmt.format(y), va="center", ha="left", fontsize=8.5, color=TEXT)


MIN_STEP = 4000  # the early steep descent (CE 4.4 -> 3.3 by step 4000) would hide the 0.02-level arm differences


def plot_set(series, key, title, fname, ylabel="CE loss (nats/token)"):
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=110)
    finals = []
    for arm in ("standard", "uniform", "beta2", "beta4"):
        pts = [(st, ce) for st, ce in series[arm][key] if st >= MIN_STEP]
        if not pts:
            continue
        x = np.array([st for st, _ in pts]) * TOKENS_PER_STEP / 1e9
        y = np.array([ce for _, ce in pts])
        a = ARMS[arm]
        ax.plot(
            x,
            y,
            a["ls"],
            color=a["color"],
            lw=2,
            solid_capstyle="round",
            label=a["label"],
            zorder=3,
        )
        ax.plot(x, y, "o", color=a["color"], ms=5, mec=SURFACE, mew=1.5, zorder=4)
        finals.append((x[-1], y[-1], a["color"]))
    ax.set_title(title, loc="left")
    style_axes(ax, ylabel)
    ax.set_xlim(MIN_STEP * TOKENS_PER_STEP / 1e9 - 0.2, 11.2)
    end_labels(ax, finals)
    legend_below(ax)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


def plot_delta(series):
    """EMO arms minus the standard router at the five permanent checkpoints, mean over the 11 sets."""
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=110)
    base = dict(series["standard"]["mean"])
    finals = []
    for arm in ("uniform", "beta2", "beta4"):
        pts = [(st, ce - base[st]) for st, ce in series[arm]["mean"] if st in base]
        x = np.array([st for st, _ in pts]) * TOKENS_PER_STEP / 1e9
        y = np.array([d for _, d in pts])
        a = ARMS[arm]
        ax.plot(x, y, "-", color=a["color"], lw=2, label=a["label"], zorder=3)
        ax.plot(x, y, "o", color=a["color"], ms=5, mec=SURFACE, mew=1.5, zorder=4)
        finals.append((x[-1], y[-1], a["color"]))
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=2)
    ax.set_title("Mean over the 11 sets, EMO arm minus standard router", loc="left")
    style_axes(ax, "Δ CE vs standard (nats/token)")
    ax.set_xlim(2, 11.2)
    end_labels(ax, finals, fmt="{:+.3f}")
    legend_below(ax, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "val_delta.png")
    plt.close(fig)


def smooth(steps, vals, window):
    v = np.asarray(vals, dtype=float)
    k = np.ones(window) / window
    sm = np.convolve(v, k, mode="valid")
    return np.asarray(steps)[window - 1 :], sm


def plot_train(curves):
    window = 100  # logged every 10 steps -> ~1000-step rolling mean
    for fname, xlim, ylim, title in (
        (
            "train_ce.png",
            (0, 10.0),
            None,
            f"Pretraining CE loss ({window * 10:,}-step rolling mean)",
        ),
        (
            "train_ce_zoom.png",
            (4.0, 10.0),
            None,
            f"Pretraining CE loss, last 6B tokens ({window * 10:,}-step rolling mean)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=110)
        finals = []
        for arm in ("standard", "uniform", "beta2", "beta4"):
            st, ce = zip(*curves[arm]["train_ce"])
            xs, ys = smooth(st, ce, window)
            x = xs * TOKENS_PER_STEP / 1e9
            m = (x >= xlim[0]) & (x <= xlim[1])
            a = ARMS[arm]
            ax.plot(x[m], ys[m], a["ls"], color=a["color"], lw=1.6, label=a["label"], zorder=3)
            finals.append((x[m][-1], ys[m][-1], a["color"]))
        ax.set_title(title, loc="left")
        style_axes(ax)
        ax.set_xlim(xlim[0], xlim[1] + 1.2)
        if fname == "train_ce_zoom.png":
            ys_all = [y for _, y, _ in finals]
            ax.set_ylim(min(ys_all) - 0.08, 2.62)
        end_labels(ax, finals)
        legend_below(ax)
        fig.tight_layout()
        fig.savefig(OUT / fname)
        plt.close(fig)

    # smoothed EMO-minus-standard curves: the four runs see the same data order, so the per-step
    # noise is common-mode and the difference is far cleaner than the raw curves
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=110)
    base_steps, base_ce = zip(*curves["standard"]["train_ce"])
    base = dict(zip(base_steps, base_ce))
    finals = []
    for arm in ("uniform", "beta2", "beta4"):
        pts = [(st, ce - base[st]) for st, ce in curves[arm]["train_ce"] if st in base]
        st, d = zip(*pts)
        xs, ys = smooth(st, d, window)
        x = xs * TOKENS_PER_STEP / 1e9
        a = ARMS[arm]
        ax.plot(x, ys, "-", color=a["color"], lw=1.6, label=a["label"], zorder=3)
        finals.append((x[-1], ys[-1], a["color"]))
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=2)
    ax.set_title(f"Pretraining CE, EMO arm minus standard ({window * 10:,}-step mean)", loc="left")
    style_axes(ax, "Δ CE vs standard (nats/token)")
    ax.set_xlim(1.0, 11.2)  # the warmup transient (steps < 2000, swings of -0.07..+0.02) is cropped
    ax.set_ylim(-0.015, 0.02)
    end_labels(ax, finals, fmt="{:+.3f}")
    legend_below(ax, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "train_ce_delta.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    series = load()
    for s in SETS:
        plot_set(series, s, f"{s} validation", f"val_{s}.png")
    plot_set(series, "mean", "Mean over the 11 validation sets", "val_mean.png")
    plot_delta(series)
    plot_train(json.load(open(IN / "train_ce_curves.json")))
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
