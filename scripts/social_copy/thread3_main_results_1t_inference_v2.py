#!/usr/bin/env python3
"""Thread 3 figure V2: 1T main-results bar chart, inference only, no caption.

Same panel content as ``thread3_main_results_1t_inference.py`` but with the
suptitle/subtitle stripped — just the three panels and their legend boxes,
ready to be paired with separate caption text in the social post.

Output: claude_outputs/social_post/thread3_main_results_1t_inference_v2.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGS_DIR = REPO_ROOT / "scripts" / "pruning_plots" / "paper_figure_codes"
sys.path.insert(0, str(PAPER_FIGS_DIR))

from plot_main_results_abs_bars import (  # noqa: E402
    FLEX_PALETTE,
    KEEPK_VALUES,
    PANEL_TITLES,
    PANEL_Y_CONFIG,
    REG_PALETTE,
    SCALES,
    _build_clumps,
    _draw_panel,
    _shade_label,
)

INPUT_CSV = REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"
OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "social_post"
OUTPUT = OUTPUT_DIR / "thread3_main_results_1t_inference_v2.png"


def _draw_legend(fig) -> None:
    keepk_labels = [_shade_label(k) for k in KEEPK_VALUES]
    common_kw = dict(
        frameon=True,
        fontsize=10,
        title_fontsize=12,
        handletextpad=0.4,
        columnspacing=0.8,
        handlelength=1.3,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
    )

    reg_handles = [
        mpatches.Patch(facecolor=REG_PALETTE[i], edgecolor="black", linewidth=0.6)
        for i in range(len(KEEPK_VALUES))
    ]
    leg_r = fig.legend(
        reg_handles,
        keepk_labels,
        title="Reg. MoE (1T tokens)",
        ncol=len(reg_handles),
        loc="lower center",
        bbox_to_anchor=(0.30, 0.005),
        facecolor="#EEF6E4",
        edgecolor=REG_PALETTE[1],
        **common_kw,
    )
    leg_r.get_title().set_fontweight("bold")
    leg_r.get_title().set_color(REG_PALETTE[0])
    leg_r.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_r)

    flex_handles = [
        mpatches.Patch(facecolor=FLEX_PALETTE[i], edgecolor="black", linewidth=0.6)
        for i in range(len(KEEPK_VALUES))
    ]
    leg_f = fig.legend(
        flex_handles,
        keepk_labels,
        title="EMO (1T tokens)",
        ncol=len(flex_handles),
        loc="lower center",
        bbox_to_anchor=(0.70, 0.005),
        facecolor="#FBE7EF",
        edgecolor=FLEX_PALETTE[2],
        **common_kw,
    )
    leg_f.get_title().set_fontweight("bold")
    leg_f.get_title().set_color(FLEX_PALETTE[0])
    leg_f.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_f)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.read_csv(INPUT_CSV)
    spec = SCALES["1t"]
    reg_model = spec["reg_model"]
    flex_model = spec["flex_model"]
    refs = spec["refs"]  # empty for 1T

    n_cols = len(PANEL_TITLES)
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 4.6))

    handles_for_legend: Dict[str, plt.Artist] = {}
    suffix = "inf"
    for c, panel_title in enumerate(PANEL_TITLES):
        ax = axes[c]
        metric_col = f"{panel_title} ({suffix})"
        clumps, overlays = _build_clumps(
            df, metric_col, reg_model, flex_model, refs
        )
        panel_cfg = {
            **PANEL_Y_CONFIG.get(panel_title, {}),
            **PANEL_Y_CONFIG.get(metric_col, {}),
        }
        _draw_panel(ax, clumps, overlays, panel_cfg, handles_for_legend)
        ax.set_title(panel_title)
        ax.set_xlabel("")
        ax.set_ylabel("Performance" if c == 0 else "")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.xaxis.grid(False)

    _draw_legend(fig)

    # Bottom margin reserved for the legend, top tight against the panels.
    fig.tight_layout(rect=(0, 0.15, 1, 1.0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
