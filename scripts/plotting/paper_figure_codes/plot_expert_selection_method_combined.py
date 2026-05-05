#!/usr/bin/env python3
"""Combined-models version of the expert-selection-method figure.

1x3 grid (MMLU, MMLU Pro, GSM8K). Each panel shows six lines — two models
(Reg. MoE, EMO) crossed with three selection methods (Random, Router,
Easy-EP). Color encodes model (green = Reg. MoE, pink = EMO);
linestyle + marker encode method.

Outputs:
    claude_outputs/prune_plots/expert_selection_method_ckpt0_combined.pdf
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
    REPO_ROOT / "claude_outputs" / "prune_plots" / "expert_selection_method_ckpt0.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "claude_outputs" / "prune_plots"
    / "expert_selection_method_ckpt0_combined.pdf"
)

PANEL_TITLES = ["MMLU", "MMLU Pro", "GSM8K"]
KEEPK_VALUES = [8, 16, 32, 64, 128]

# (display label, model name in CSV, color)
MODELS: List[Tuple[str, str, str]] = [
    ("Reg. MoE", "Reg. MoE", "#5B8E3F"),  # medium green (matches abs_bars REG_PALETTE[1])
    ("EMO", "FlexMoE", "#B8327C"),     # magenta-pink (matches abs_bars FLEX_PALETTE[2])
]

# (display label, csv column suffix, linestyle, marker)
METHODS: List[Tuple[str, str, str, str]] = [
    ("Random",  "Random",  ":",  "X"),
    ("Router",  "Router",  "-",  "o"),
    ("Easy-EP", "Easy-EP", "--", "s"),
]

PANEL_Y_CONFIG: Dict[str, Dict[str, float]] = {
    "MMLU": {"ymin": 20.0, "random_chance": 25.0},
    "MMLU Pro": {"ymin": 7.0, "random_chance": 10.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _parse_experts(value) -> int:
    return int(str(value).split()[0])


def _series(
    df: pd.DataFrame, model_name: str, panel_title: str, method_suffix: str
) -> Tuple[List[int], List[float]]:
    sub = df[df.iloc[:, 0] == model_name].copy()
    if sub.empty:
        raise KeyError(f"No rows for model {model_name!r}")
    col = f"{panel_title} {method_suffix}"
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")
    sub["x"] = sub["# Total Experts"].map(_parse_experts)
    sub = sub.sort_values("x")
    xs: List[int] = []
    ys: List[float] = []
    for _, row in sub.iterrows():
        v = row[col]
        if pd.isna(v):
            continue
        xs.append(int(row["x"]))
        ys.append(float(v))
    return xs, ys


def _draw_panel(ax, df: pd.DataFrame, panel_title: str,
                panel_cfg: Dict[str, float]) -> None:
    all_ys: List[float] = []
    for model_label, model_name, color in MODELS:
        for method_label, suffix, linestyle, marker in METHODS:
            xs, ys = _series(df, model_name, panel_title, suffix)
            all_ys.extend(ys)
            ax.plot(
                xs, ys,
                color=color, linestyle=linestyle, marker=marker,
                linewidth=3.0, markersize=10,
                markeredgecolor="black", markeredgewidth=0.6,
                alpha=0.95,
                clip_on=False,  # don't clip markers that sit at the axis bound
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(KEEPK_VALUES)
    ax.set_xticklabels([str(k) for k in KEEPK_VALUES])
    # 128 on the left, 8 on the right.
    ax.set_xlim(KEEPK_VALUES[-1] * 1.06, KEEPK_VALUES[0] * 0.94)

    # Y-limits: respect the configured ymin but extend a touch below the
    # lowest plotted value so a marker exactly at ymin (e.g. GSM8K Random=0)
    # still renders fully inside the visible axes area.
    ymin_config = float(panel_cfg.get("ymin", 0.0))
    if all_ys:
        y_min_data = min(all_ys)
        y_max_data = max(all_ys)
        y_range = max(y_max_data - y_min_data, 1.0)
        pad_below = y_range * 0.04
        effective_ymin = min(ymin_config, y_min_data - pad_below)
        ax.set_ylim(bottom=effective_ymin)
    else:
        ax.set_ylim(bottom=ymin_config)

    chance = panel_cfg.get("random_chance")
    if chance is not None:
        ax.axhline(
            chance, color="#222222", linewidth=1.4,
            linestyle=(0, (4, 3)), zorder=1, alpha=0.7,
        )
        x_right = ax.get_xlim()[1]
        ax.text(
            x_right, chance, "  random",
            ha="left", va="center", fontsize=8.5, color="#222222",
            fontweight="bold", clip_on=False, zorder=1,
        )


def _draw_legend(fig) -> None:
    """Two centered tinted boxes — one per model. Each box lists the three
    selection methods as line entries drawn in that model's color, matching
    the aesthetic used in main_results_abs_bars_1t.pdf.
    """
    common_kw = dict(
        frameon=True,
        fontsize=14,
        title_fontsize=15,
        handletextpad=0.7,
        columnspacing=1.2,
        borderpad=0.9,
        labelspacing=0.5,
        handlelength=3.2,
    )

    method_labels = [m[0] for m in METHODS]

    # Reg. MoE box (left of center) — green wash + green border.
    reg_color = MODELS[0][2]
    reg_handles = [
        mlines.Line2D(
            [], [], color=reg_color, linestyle=linestyle, marker=marker,
            linewidth=3.0, markersize=10,
            markeredgecolor="black", markeredgewidth=0.6,
        )
        for _label, _suffix, linestyle, marker in METHODS
    ]
    leg_r = fig.legend(
        reg_handles, method_labels,
        title="Reg. MoE",
        ncol=len(METHODS),
        loc="lower center", bbox_to_anchor=(0.32, 0.005),
        facecolor="#EEF6E4", edgecolor=reg_color,
        **common_kw,
    )
    leg_r.get_title().set_fontweight("bold")
    leg_r.get_title().set_color("#225C2E")
    leg_r.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_r)

    # EMO box (right of center) — pink wash + pink border.
    flex_color = MODELS[1][2]
    flex_handles = [
        mlines.Line2D(
            [], [], color=flex_color, linestyle=linestyle, marker=marker,
            linewidth=3.0, markersize=10,
            markeredgecolor="black", markeredgewidth=0.6,
        )
        for _label, _suffix, linestyle, marker in METHODS
    ]
    leg_f = fig.legend(
        flex_handles, method_labels,
        title="EMO",
        ncol=len(METHODS),
        loc="lower center", bbox_to_anchor=(0.68, 0.005),
        facecolor="#FBE7EF", edgecolor=flex_color,
        **common_kw,
    )
    leg_f.get_title().set_fontweight("bold")
    leg_f.get_title().set_color("#3F1052")
    leg_f.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_f)


def render_figure(df: pd.DataFrame, output_path: Path) -> None:
    n_cols = len(PANEL_TITLES)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(16, 5.8),
    )

    for c, panel_title in enumerate(PANEL_TITLES):
        ax = axes[c]
        panel_cfg = PANEL_Y_CONFIG.get(panel_title, {})
        _draw_panel(ax, df, panel_title, panel_cfg)
        ax.set_title(panel_title)
        ax.set_xlabel("# total experts")
        ax.set_ylabel("Performance" if c == 0 else "")
        ax.grid(True, axis="y", alpha=0.7)
        ax.grid(False, axis="x")

    _draw_legend(fig)
    fig.tight_layout(rect=(0, 0.17, 1, 1.0))

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
