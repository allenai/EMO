#!/usr/bin/env python3
"""Few-shot importance figure as a clustered scatter (no connecting lines).

1x3 figure (MMLU, MMLU Pro, GSM8K). Per panel:
    - X-axis is "binned" by keepk in {8, 16, 32, 128}.
    - Inside each bin, every (few-shot config, n) point is plotted as its
      own marker. n values 1, 5, 10, 100, All are spread left-to-right
      within the bin (n=1 leftmost, n=All rightmost), with the three
      few-shot configs slightly offset around each n sub-position.
    - Color encodes the few-shot config; marker shape encodes n.

Note on GSM8K seed averaging:
    For GSM8K, each (config, keepk, n ∈ {1, 5, 10}) point is the mean across
    three pruning-calibration seeds (seed-0 + `_pseed-1` + `_pseed-2` dirs).
    The averaging happens upstream in
    `scripts/pruning_plots/get_table_scores_nprune_ablation.py`, so
    this script only needs to consume the validation_sample_ablation_ckpt0.csv
    and plot the cell value as-is. All other points (other tasks, larger n,
    keepk=128) are unchanged seed-0 values.

Outputs:
    claude_outputs/prune_plots/validation_fewshot_importance_clustered_ckpt0.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "validation_sample_ablation_ckpt0.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "claude_outputs" / "prune_plots"
    / "validation_fewshot_importance_clustered_ckpt0.pdf"
)

KEEPK_VALUES = [128, 32, 16, 8]  # bin order, left -> right
TASKS = ["MMLU", "MMLU Pro", "GSM8K"]

# (display label, prunemode in CSV, color). Picked to be cohesive with the
# rest of the paper figures (EMO pink, Dense-baseline orange, weak gray)
# while staying clearly separable from each other.
CONFIGS: List[Tuple[str, str, str]] = [
    ("few-shot select + few-shot eval", "Router",          "#B8327C"),  # EMO magenta
    ("few-shot select + 0-shot eval",   "Router (e0)",     "#E78532"),  # Dense-baseline orange
    ("0-shot select + 0-shot eval",     "Router (0-shot)", "#888888"),  # neutral gray
]

# (display label, csv n token, marker shape). Order = sub-position order
# within each bin, left -> right.
N_LEVELS: List[Tuple[str, str, str]] = [
    ("All", "All", "o"),  # circle  (leftmost: largest n)
    ("100", "100", "^"),  # up triangle
    ("10",  "10",  "D"),  # diamond
    ("5",   "5",   "s"),  # square
    ("1",   "1",   "v"),  # down triangle (rightmost: smallest n)
]

PANEL_Y_CONFIG: Dict[str, Dict[str, float]] = {
    "MMLU": {"ymin": 22.0},
    "MMLU Pro": {"ymin": 8.0},
}

# Within each keepk "bin", spread the 5 n sub-positions across this fraction
# of unit spacing, and the 3 configs across this tiny inner offset.
BIN_HALF_WIDTH = 0.34
CONFIG_OFFSET = 0.045


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _value(
    df: pd.DataFrame, prunemode: str, task: str, keepk: int, n_label: str
) -> Optional[float]:
    sub = df[(df["Model"] == "FlexMoE") & (df["Prunemode"] == prunemode)
             & (df["Task"] == task)]
    if sub.empty:
        return None
    col = f"{keepk} Experts ({n_label})"
    if col not in df.columns:
        if keepk == 128:
            col = "128 Experts (All)"
            if col not in df.columns:
                return None
        else:
            return None
    v = sub.iloc[0][col]
    return None if pd.isna(v) else float(v)


def _n_offsets(n_levels_count: int) -> List[float]:
    """Evenly-spaced sub-positions for n values within a bin, from -BIN_HALF_WIDTH
    on the left to +BIN_HALF_WIDTH on the right."""
    if n_levels_count == 1:
        return [0.0]
    step = (2 * BIN_HALF_WIDTH) / (n_levels_count - 1)
    return [-BIN_HALF_WIDTH + i * step for i in range(n_levels_count)]


def _config_inner_offsets(n_configs: int) -> List[float]:
    """Tiny offsets so the configs at the same n sub-position are not exactly
    on top of each other."""
    if n_configs == 1:
        return [0.0]
    center = (n_configs - 1) / 2
    return [(i - center) * CONFIG_OFFSET for i in range(n_configs)]


def _draw_panel(ax, df: pd.DataFrame, task: str,
                panel_cfg: Dict[str, float]) -> None:
    bin_centers = list(range(len(KEEPK_VALUES)))
    n_offsets = _n_offsets(len(N_LEVELS))
    cfg_offsets = _config_inner_offsets(len(CONFIGS))
    all_ys: List[float] = []

    for n_idx, (_n_label, n_token, marker) in enumerate(N_LEVELS):
        for c_idx, (_cfg_label, prunemode, color) in enumerate(CONFIGS):
            xs: List[float] = []
            ys: List[float] = []
            for b_idx, keepk in enumerate(KEEPK_VALUES):
                v = _value(df, prunemode, task, keepk, n_token)
                if v is None:
                    continue
                xs.append(bin_centers[b_idx] + n_offsets[n_idx] + cfg_offsets[c_idx])
                ys.append(v)
                all_ys.append(v)
            if not xs:
                continue
            ax.scatter(
                xs, ys,
                color=color, marker=marker, s=85,
                edgecolor="black", linewidth=0.6, zorder=3,
                clip_on=False,
            )

    # Light vertical separator between bins for readability.
    for b_idx in range(len(KEEPK_VALUES) - 1):
        ax.axvline(
            b_idx + 0.5, color="#cccccc", linewidth=0.8,
            linestyle="-", zorder=1,
        )

    ax.set_xticks(bin_centers)
    ax.set_xticklabels([str(k) for k in KEEPK_VALUES])
    ax.set_xlim(-0.5, len(KEEPK_VALUES) - 0.5)

    ymin_cfg = float(panel_cfg.get("ymin", 0.0))
    if all_ys:
        y_min_data = min(all_ys)
        y_max_data = max(all_ys)
        y_range = max(y_max_data - y_min_data, 1.0)
        pad = y_range * 0.10
        effective_ymin = min(ymin_cfg, y_min_data - pad)
        ax.set_ylim(bottom=effective_ymin, top=y_max_data + pad)


def _draw_legend(fig) -> None:
    common_kw = dict(
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        handletextpad=0.6,
        columnspacing=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        handlelength=1.6,
    )

    # Config legend (color), drawn with neutral circles.
    cfg_handles = [
        mlines.Line2D(
            [], [], color=color, marker="o", linestyle="None", markersize=10,
            markeredgecolor="black", markeredgewidth=0.6, label=lbl,
        )
        for lbl, _pm, color in CONFIGS
    ]
    leg_c = fig.legend(
        cfg_handles, [h.get_label() for h in cfg_handles],
        title="Few-shot configuration",
        ncol=1,
        loc="lower center", bbox_to_anchor=(0.32, 0.005),
        facecolor="#F5F5F5", edgecolor="#888888",
        **common_kw,
    )
    leg_c.get_title().set_fontweight("bold")
    leg_c.get_frame().set_linewidth(1.0)
    fig.add_artist(leg_c)

    # n legend (marker shape), drawn in neutral gray to focus on shape.
    n_handles = [
        mlines.Line2D(
            [], [], color="#444444", marker=marker, linestyle="None", markersize=10,
            markeredgecolor="black", markeredgewidth=0.6, label=lbl,
        )
        for lbl, _tok, marker in N_LEVELS
    ]
    leg_n = fig.legend(
        n_handles, [h.get_label() for h in n_handles],
        title="n (calibration samples)",
        ncol=len(n_handles),
        loc="lower center", bbox_to_anchor=(0.70, 0.005),
        facecolor="#F5F5F5", edgecolor="#888888",
        **common_kw,
    )
    leg_n.get_title().set_fontweight("bold")
    leg_n.get_frame().set_linewidth(1.0)
    fig.add_artist(leg_n)


def render_figure(df: pd.DataFrame, output_path: Path) -> None:
    n_cols = len(TASKS)
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 5.4))
    for c, task in enumerate(TASKS):
        ax = axes[c]
        panel_cfg = PANEL_Y_CONFIG.get(task, {})
        _draw_panel(ax, df, task, panel_cfg)
        ax.set_title(task)
        ax.set_xlabel("# total experts")
        ax.set_ylabel("Performance" if c == 0 else "")
        ax.grid(True, axis="y", alpha=0.7)
        ax.grid(False, axis="x")

    _draw_legend(fig)
    fig.tight_layout(rect=(0, 0.20, 1, 1.0))

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
