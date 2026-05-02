#!/usr/bin/env python3
"""ModMoE (two-level batch LB) LR ablations, first 3000 steps.

Single-panel sibling of ``plot_moe_hyperparameter_ablations.py``. Plots
``train/CE loss`` for the three ModMoE runs at lb=1e-1 with different
learning rates:

    lr=4e-2  (high)
    lr=4e-3  (medium, the preferred config)
    lr=4e-4  (low — sparser logging in the early steps)

Color scheme matches the rest of the paper figures: a magenta/pink
gradient on the FlexMoE/ModMoE family, with the lr=4e-3 line at the
saturated brand magenta as the "preferred config" highlight.

Reads : claude_outputs/other_figures/twolevel_hyperparameter_ablations.csv
Writes: claude_outputs/other_figures/twolevel_hyperparameter_ablations.png
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
    REPO_ROOT / "claude_outputs" / "other_figures"
    / "twolevel_hyperparameter_ablations.csv"
)
OUT_PATH = (
    REPO_ROOT / "claude_outputs" / "other_figures"
    / "twolevel_hyperparameter_ablations.png"
)

METRIC_SUFFIX = " - train/CE loss"
MAX_STEP = 3000

# (display label, wandb run name, color). All runs share lb=1e-1.
RUNS: List[Tuple[str, str, str]] = [
    (
        "lr=4e-2",
        "twolevelbatchlb-32_1b14b_lr-4e-2_lb-1e-1_ablations_0117",
        "#3F1052",  # deep purple
    ),
    (
        "lr=4e-3",
        "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119",
        "#B8327C",  # ModMoE magenta (preferred)
    ),
    (
        "lr=4e-4",
        "twolevelbatchlb-32_1b14b_lr-4e-4_lb-1e-1_0118",
        "#E48AB5",  # light pink
    ),
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

    ax.set_title("ModMoE: learning rate ablations", fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("train CE loss")
    ax.set_xlim(0, MAX_STEP)

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
        title="ModMoE Hyperparameters",
        ncol=len(handles),
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        frameon=True, fontsize=11, title_fontsize=12,
        handletextpad=0.6, columnspacing=1.2,
        borderpad=0.7, labelspacing=0.4, handlelength=2.4,
        facecolor="#FBE7EF", edgecolor="#B8327C",  # ModMoE pink-tinted frame
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#3F1052")
    leg.get_frame().set_linewidth(1.1)

    fig.tight_layout(rect=(0, 0.13, 1, 1.0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
