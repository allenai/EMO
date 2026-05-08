#!/usr/bin/env python3
"""Thread 7 figure: 130B Pareto plot with social caption.

Re-renders the 130B MMLU Pareto plot using helpers from
``scripts/pruning_plots/paper_figure_codes/plot_main_results_abs_bars_130b_shrunken.py``,
with a baked-in title caption that ties it to the memory-budget message.

Output: claude_outputs/social_post/thread7_pareto_130b.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGS_DIR = REPO_ROOT / "scripts" / "pruning_plots" / "paper_figure_codes"
sys.path.insert(0, str(PAPER_FIGS_DIR))

from plot_main_results_abs_bars import (  # noqa: E402
    DENSE_COLOR,
    FLEX_PALETTE,
    PANEL_Y_CONFIG,
    REG_PALETTE,
    REGSMALL_COLOR,
    SCALES,
)
from plot_main_results_abs_bars_130b_shrunken import (  # noqa: E402
    DENSE_MARKER,
    FLEX_LINE_COLOR,
    FLEX_MARKER,
    MODE_SUFFIX,
    PANEL_TITLE,
    REG_LINE_COLOR,
    REG_MARKER,
    REGSMALL_MARKER,
    _draw_panel,
)

INPUT_CSV = REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"
OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "social_post"
OUTPUT = OUTPUT_DIR / "thread7_pareto_130b.png"

HEADER_TITLE = (
    "EMO pushes the Pareto frontier in memory-accuracy trade-off"
)
# Three-line subtitle: each line is a separate clarification, rendered
# stacked under the title so they don't overflow the panel width.
HEADER_SUBTITLE_LINES = (
    "EMO and Reg. MoE: a single model evaluated across different expert subset sizes.",
    "Dense @8 = dense model trained from scratch for a fixed 8-expert budget.",
    "Reg. MoE @32 = standard MoE trained from scratch for a fixed 32-expert budget.",
)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(INPUT_CSV)

    spec = SCALES["130b"]
    flex_model = spec["flex_model"]
    reg_model = spec["reg_model"]
    refs = spec["refs"]

    metric_col = f"{PANEL_TITLE} ({MODE_SUFFIX})"
    panel_cfg = {
        **PANEL_Y_CONFIG.get(PANEL_TITLE, {}),
        **PANEL_Y_CONFIG.get(metric_col, {}),
    }
    random_chance = panel_cfg.get("random_chance")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    _draw_panel(
        ax, df, metric_col, flex_model, reg_model, refs,
        random_chance=random_chance,
    )

    fig.suptitle(
        HEADER_TITLE,
        x=0.5, y=0.995,
        fontsize=14, fontweight="bold", color="#1a202c",
    )
    # Stacked subtitle lines: top->down so they read like a legend caption.
    line_y_start = 0.945
    line_step = 0.030
    for i, line in enumerate(HEADER_SUBTITLE_LINES):
        fig.text(
            0.5, line_y_start - i * line_step,
            line,
            ha="center", va="top",
            fontsize=10, color="#556070",
        )

    ax.set_title("")
    ax.set_xlabel("Expert subset size (memory budget)", fontsize=11)
    ax.set_ylabel("MMLU accuracy", fontsize=11)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        mlines.Line2D(
            [], [], color=FLEX_LINE_COLOR, linewidth=2.6,
            marker=FLEX_MARKER, markersize=9,
            markerfacecolor=FLEX_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.2,
            label="EMO",
        ),
        mlines.Line2D(
            [], [], color=REG_LINE_COLOR, linewidth=2.4,
            marker=REG_MARKER, markersize=8,
            markerfacecolor=REG_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.0,
            label="Reg. MoE",
        ),
        mlines.Line2D(
            [], [], color="none", marker=DENSE_MARKER, markersize=14,
            markerfacecolor=DENSE_COLOR,
            markeredgecolor="black", markeredgewidth=0.8,
            label="Dense @8 (trained from scratch)",
        ),
        mlines.Line2D(
            [], [], color="none", marker=REGSMALL_MARKER, markersize=10,
            markerfacecolor=REGSMALL_COLOR,
            markeredgecolor="black", markeredgewidth=0.8,
            label="Reg. MoE @32 (trained from scratch)",
        ),
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        frameon=True, fontsize=9,
        handletextpad=0.5, borderpad=0.6,
        facecolor="white", edgecolor="#CCCCCC",
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=(0, 0.0, 1, 0.84))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")

    # Reference palettes so the linter doesn't complain about unused imports.
    _ = (FLEX_PALETTE, REG_PALETTE)


if __name__ == "__main__":
    main()
