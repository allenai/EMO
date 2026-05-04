#!/usr/bin/env python3
"""Architecture-norm ablation: Prenorm + No QK-norm vs. ReorderedNorm,
crossed with model family (Standard MoE vs. EMO), first 3000 steps.

Single-panel sibling of ``plot_moe_hyperparameter_ablations.py`` and
``plot_twolevel_hyperparameter_ablations.py``. Plots ``train/CE loss``
for the 2x2 cross of (family, norm config):

    Standard MoE | Prenorm + No QK-norm
    Standard MoE | ReorderedNorm
    EMO       | Prenorm + No QK-norm
    EMO       | ReorderedNorm

Color encodes family (green = Standard MoE / Reg. MoE, magenta = EMO),
linestyle encodes the norm config (solid = Prenorm + No QK-norm, dashed
= ReorderedNorm).

Reads : claude_outputs/other_figures/prenorm_noqknorm_ablations.csv
Writes: claude_outputs/other_figures/prenorm_noqknorm_ablations.pdf
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
    / "prenorm_noqknorm_ablations.csv"
)
OUT_PATH = (
    REPO_ROOT / "claude_outputs" / "other_figures"
    / "prenorm_noqknorm_ablations.pdf"
)

METRIC_SUFFIX = " - train/CE loss"
MAX_STEP = 3000

# Four distinct colors: greens for Standard MoE, pinks for EMO.
# Within each family the darker shade = Prenorm + No QK-norm; the lighter
# shade = ReorderedNorm. All four lines are solid so color/shade alone
# carries the family-and-norm distinction.
RUNS: List[Tuple[str, str, str, str]] = [
    (
        "Standard MoE, Prenorm + No QK-norm",
        "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123",
        "#225C2E", "-",  # dark green
    ),
    (
        "Standard MoE, ReorderedNorm",
        "moe_1b14b_128experts_olmoe-mix_130B_1117",
        "#93C265", "-",  # light green
    ),
    (
        "EMO, Prenorm + No QK-norm",
        "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121",
        "#6E1F73", "-",  # deep purple
    ),
    (
        "EMO, ReorderedNorm",
        "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115",
        "#E48AB5", "-",  # light pink
    ),
]


def load_long(df: pd.DataFrame, runs: List[Tuple[str, str, str, str]],
              max_step: int) -> pd.DataFrame:
    frames = []
    for label, run, _color, _ls in runs:
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

    palette: Dict[str, str] = {label: color for label, _run, color, _ls in RUNS}
    dashes: Dict[str, Tuple] = {
        label: (1, 0) if ls == "-" else (5, 2)
        for label, _run, _color, ls in RUNS
    }
    hue_order = [label for label, _run, _color, _ls in RUNS]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.lineplot(
        data=long_df,
        x="step", y="ce_loss",
        hue="config", hue_order=hue_order,
        style="config", style_order=hue_order,
        palette=palette,
        dashes=dashes,
        linewidth=2.5,
        ax=ax,
    )

    ax.set_title(
        "Architecture × Norm ablation",
        fontweight="bold",
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("train CE loss")
    ax.set_xlim(0, MAX_STEP)
    ax.set_ylim(top=5.0)

    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles = [
        mlines.Line2D(
            [], [], color=color, linestyle=ls, linewidth=2.8, label=label,
        )
        for label, _run, color, ls in RUNS
    ]
    leg = fig.legend(
        handles, [h.get_label() for h in handles],
        title="Family,  Norm",
        ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        frameon=True, fontsize=11, title_fontsize=12,
        handletextpad=0.6, columnspacing=1.4,
        borderpad=0.7, labelspacing=0.4, handlelength=2.6,
        facecolor="#F5F5F5", edgecolor="#888888",
    )
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_linewidth(1.0)

    fig.tight_layout(rect=(0, 0.18, 1, 1.0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
