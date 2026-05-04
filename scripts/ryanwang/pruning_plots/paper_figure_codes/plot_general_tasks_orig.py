#!/usr/bin/env python3
"""Visualize ``general_tasks_orig.csv`` (no-finetune full-model evals) as
a 1x5 grouped bar figure.

Five task panels (MC9, Gen5, MMLU, MMLU Pro, GSM8K). Each panel has up
to three training-scale clumps (5T, 1T, 130B). Inside each clump, one
bar per available model:

    5T:    OLMoE†                                 (orange)
    1T:    Reg. MoE,  EMO (Ours)               (green, magenta)
    130B:  Dense, Reg. MoE,  EMO (Ours)        (gray, green, magenta)

Color scheme matches the rest of the paper figures (Reg. MoE green,
EMO magenta, Dense neutral gray, OLMoE warm orange for the external
baseline).

Outputs:
    claude_outputs/prune_plots/general_tasks_orig.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO_ROOT / "claude_outputs" / "prune_plots" / "general_tasks_orig.csv"
DEFAULT_OUTPUT = REPO_ROOT / "claude_outputs" / "prune_plots" / "general_tasks_orig.pdf"

TASKS = ["MC9", "Gen5", "MMLU", "MMLU Pro", "GSM8K"]

# Per-model color (from the paper-figure palette).
MODEL_COLOR: Dict[str, str] = {
    "OLMoE^dagger":   "#E78532",  # warm orange — external baseline
    "Dense":          "#888888",  # gray — vanilla dense baseline
    "Reg. MoE":       "#5B8E3F",  # green — same as abs_bars REG_PALETTE[1]
    "FlexMoE (Ours)": "#B8327C",  # magenta — same as EMO in abs_bars
}

# Display name for the legend.
MODEL_DISPLAY: Dict[str, str] = {
    "OLMoE^dagger":   "OLMoE†",
    "Dense":          "Dense",
    "Reg. MoE":       "Reg. MoE",
    "FlexMoE (Ours)": "EMO (Ours)",
}

# Scale clumps in display order (left -> right within each panel).
# Each entry: (scale_label, [model_keys to render in that clump]).
SCALES: List[Tuple[str, List[str]]] = [
    ("5T",   ["OLMoE^dagger"]),
    ("1T",   ["Reg. MoE", "FlexMoE (Ours)"]),
    ("130B", ["Dense", "Reg. MoE", "FlexMoE (Ours)"]),
]

# Layout knobs.
BAR_WIDTH = 0.8
INTRA_BAR_STEP = 1.0
CLUMP_GAP = 1.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _value(df: pd.DataFrame, model: str, scale: str, task: str) -> Optional[float]:
    name_col = df.columns[0]
    sub = df[(df[name_col] == model) & (df["# train tokens"] == scale)]
    if sub.empty:
        return None
    v = sub.iloc[0][task]
    if pd.isna(v) or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _draw_panel(ax, df: pd.DataFrame, task: str) -> None:
    cursor = 0.0
    clump_centers: List[float] = []
    clump_labels: List[str] = []
    all_vals: List[float] = []

    for scale_label, models in SCALES:
        positions: List[float] = []
        heights: List[float] = []
        colors: List[str] = []
        for i, model in enumerate(models):
            v = _value(df, model, scale_label, task)
            if v is None:
                continue
            x = cursor + i * INTRA_BAR_STEP
            positions.append(x)
            heights.append(v)
            colors.append(MODEL_COLOR[model])
            all_vals.append(v)

        if not positions:
            cursor += len(models) * INTRA_BAR_STEP + CLUMP_GAP
            continue

        ax.bar(
            positions, heights, BAR_WIDTH,
            color=colors, edgecolor="black", linewidth=0.6,
        )
        for x, v in zip(positions, heights):
            ax.annotate(
                f"{v:.1f}",
                xy=(x, v), xytext=(0, 3),
                textcoords="offset points", ha="center",
                fontsize=9, color="#222222",
            )

        clump_start = positions[0]
        clump_end = positions[-1]
        clump_centers.append((clump_start + clump_end) / 2)
        clump_labels.append(scale_label)
        cursor = clump_end + INTRA_BAR_STEP + CLUMP_GAP

    if all_vals:
        ymax = max(all_vals)
        ax.set_ylim(top=ymax * 1.20, bottom=0)
        y_text = -0.08 * ymax
    else:
        y_text = -0.5

    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.xaxis.grid(False)

    for cx, cl in zip(clump_centers, clump_labels):
        ax.text(
            cx, y_text, cl, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#333333",
            transform=ax.transData, clip_on=False,
        )


def _draw_legend(fig) -> None:
    common_kw = dict(
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0.7,
        labelspacing=0.4,
        handlelength=1.4,
        handleheight=1.0,
    )
    handles = [
        mpatches.Patch(
            facecolor=MODEL_COLOR[m], edgecolor="black", linewidth=0.6,
            label=MODEL_DISPLAY[m],
        )
        for m in ("OLMoE^dagger", "Dense", "Reg. MoE", "FlexMoE (Ours)")
    ]
    leg = fig.legend(
        handles, [h.get_label() for h in handles],
        title="Model",
        ncol=len(handles),
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        facecolor="#F5F5F5", edgecolor="#888888",
        **common_kw,
    )
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_linewidth(1.0)


def render_figure(df: pd.DataFrame, output_path: Path) -> None:
    n_cols = len(TASKS)
    fig, axes = plt.subplots(1, n_cols, figsize=(18, 4.5))
    for c, task in enumerate(TASKS):
        ax = axes[c]
        _draw_panel(ax, df, task)
        ax.set_title(task)
        ax.set_xlabel("")
        ax.set_ylabel("Performance" if c == 0 else "")
        ax.grid(True, axis="y", alpha=0.7)

    _draw_legend(fig)
    fig.tight_layout(rect=(0, 0.13, 1, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    df = pd.read_csv(args.input)
    render_figure(df, args.output)


if __name__ == "__main__":
    main()
