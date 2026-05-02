#!/usr/bin/env python3
"""MoE LR / load-balance hyperparameter ablations, first 5000 steps.

Single-panel grad-norm-style line plot, sibling of
``plot_modmoe_global_vs_local_lb.py``. Plots ``train/CE loss`` for four
1B-14B MoE runs:

    lr=4e-2, lb=1e-2  (high lr)
    lr=4e-3, lb=1e-2  (medium lr)
    lr=4e-4, lb=1e-2  (low lr)
    lr=4e-3, lb=1e-1  (medium lr with stronger LB loss)

Color scheme matches the rest of the paper figures: the three lb=1e-2
LR variants form a green gradient (light to dark = low to high lr) and
the lr=4e-3 / lb=1e-1 alternative gets ModMoE magenta as the "preferred
config" highlight.

Reads : claude_outputs/other_figures/moe_hyperparameter_ablations.csv
Writes: claude_outputs/other_figures/moe_hyperparameter_ablations.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = (
    REPO_ROOT / "claude_outputs" / "other_figures" / "moe_hyperparameter_ablations.csv"
)
OUT_PATH = (
    REPO_ROOT / "claude_outputs" / "other_figures" / "moe_hyperparameter_ablations.png"
)

METRIC_SUFFIX = " - train/CE loss"
MAX_STEP = 3000

# (display label, wandb run name, color)
# Order = legend order; the lb=1e-1 alternative comes last as the "winner".
RUNS: List[Tuple[str, str, str]] = [
    ("lr=4e-2, lb=1e-2", "moe_1b14b_lr-4e-2-ablations_0116",         "#225C2E"),  # dark green
    ("lr=4e-3, lb=1e-2", "moe_1b14b_lr-4e-3-ablations_0116",         "#5B8E3F"),  # medium green
    ("lr=4e-4, lb=1e-2", "moe_1b14b_lr-4e-4-ablations_0116",         "#C5DD93"),  # light green
    ("lr=4e-3, lb=1e-1", "moe_1b14b_lr-4e-3_lb-1e-1-ablations_0116", "#B8327C"),  # ModMoE magenta
]


def load_long(df: pd.DataFrame, runs: List[Tuple[str, str, str]],
              max_step: int) -> pd.DataFrame:
    frames = []
    for label, run, _color in runs:
        col = f"{run}{METRIC_SUFFIX}"
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")
        y = pd.to_numeric(df[col], errors="coerce")
        mask = y.notna() & (df["Step"] <= max_step)
        frames.append(
            pd.DataFrame(
                {
                    "step": df.loc[mask, "Step"].values,
                    "ce_loss": y[mask].values,
                    "config": label,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.read_csv(CSV_PATH)
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")

    long_df = load_long(df, RUNS, MAX_STEP)

    palette: Dict[str, str] = {label: color for label, _run, color in RUNS}
    hue_order = [label for label, _run, _color in RUNS]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.lineplot(
        data=long_df,
        x="step", y="ce_loss",
        hue="config",
        hue_order=hue_order,
        palette=palette,
        linewidth=2.5,
        ax=ax,
    )

    ax.set_title("MoE: learning rate & load-balance ablations", fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("train CE loss")
    ax.set_xlim(0, MAX_STEP)

    # Drop the auto seaborn legend; rebuild with the paper-style tinted box.
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles = [
        mlines.Line2D(
            [], [], color=palette[label], linewidth=2.8, label=label,
        )
        for label in hue_order
    ]
    leg = fig.legend(
        handles, [h.get_label() for h in handles],
        title="Standard MoE Hyperparameters",
        ncol=len(handles),
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        frameon=True, fontsize=11, title_fontsize=12,
        handletextpad=0.6, columnspacing=1.2,
        borderpad=0.7, labelspacing=0.4, handlelength=2.4,
        facecolor="#EEF6E4", edgecolor="#5B8E3F",  # green-tinted, matches Reg. MoE family
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#225C2E")
    leg.get_frame().set_linewidth(1.1)

    fig.tight_layout(rect=(0, 0.13, 1, 1.0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
