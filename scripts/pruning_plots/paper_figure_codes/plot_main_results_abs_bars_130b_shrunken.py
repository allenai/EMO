#!/usr/bin/env python3
"""Single-panel Pareto plot for 130B FlexMoE on MMLU (fine-tune).

Sweeps FlexMoE across memory regimes (keepk = 8 .. 128, log-2 x-axis)
and overlays Dense @8 and Reg. MoE @32 trained baselines as marker
points at their respective memory budgets. The story: at each budget,
FlexMoE pruned to that budget matches or beats a model that was
trained specifically for that budget.

Output:
    claude_outputs/prune_plots/main_results_130b_shrunken.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_main_results_abs_bars import (
    DENSE_COLOR,
    FLEX_PALETTE,
    KEEPK_VALUES,
    PANEL_Y_CONFIG,
    REG_PALETTE,
    REGSMALL_COLOR,
    SCALES,
    _lookup,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots"

PANEL_TITLE = "MMLU"
MODE_SUFFIX = "ft"
MODE_LABEL = "Fine-tune"

KEEPK_AXIS: List[int] = sorted(KEEPK_VALUES)

FLEX_LINE_COLOR = FLEX_PALETTE[0]
REG_LINE_COLOR = REG_PALETTE[0]
FLEX_MARKER = "o"
REG_MARKER = "s"
DENSE_MARKER = "*"
REGSMALL_MARKER = "D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "main_results_130b_shrunken.pdf",
    )
    return parser.parse_args()


def _draw_panel(
    ax,
    df: pd.DataFrame,
    metric_col: str,
    flex_model: str,
    reg_model: str,
    refs: List[Tuple[str, str, int, str]],
    random_chance: float = None,
) -> None:
    all_vals: List[float] = []

    reg_pairs = []
    for k in KEEPK_AXIS:
        v = _lookup(df, reg_model, k, metric_col)
        if v is not None:
            reg_pairs.append((k, v))
            all_vals.append(v)

    if reg_pairs:
        xs, ys = zip(*reg_pairs)
        ax.plot(
            xs, ys,
            color=REG_LINE_COLOR, linewidth=2.4,
            marker=REG_MARKER, markersize=8,
            markerfacecolor=REG_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, clip_on=False,
        )
        for x, v in zip(xs, ys):
            ax.annotate(
                f"{v:.1f}", xy=(x, v), xytext=(0, -10),
                textcoords="offset points", ha="center", va="top",
                fontsize=9.5, color=REG_LINE_COLOR, fontweight="bold",
            )

    flex_pairs = []
    for k in KEEPK_AXIS:
        v = _lookup(df, flex_model, k, metric_col)
        if v is not None:
            flex_pairs.append((k, v))
            all_vals.append(v)

    if flex_pairs:
        xs, ys = zip(*flex_pairs)
        ax.plot(
            xs, ys,
            color=FLEX_LINE_COLOR, linewidth=2.6,
            marker=FLEX_MARKER, markersize=9,
            markerfacecolor=FLEX_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.2,
            zorder=4, clip_on=False,
        )
        for x, v in zip(xs, ys):
            ax.annotate(
                f"{v:.1f}", xy=(x, v), xytext=(0, 9),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=10, color=FLEX_LINE_COLOR, fontweight="bold",
            )

    for label, model_name, k, color in refs:
        v = _lookup(df, model_name, k, metric_col, prefer_trained=True)
        if v is None:
            continue
        marker = DENSE_MARKER if "Dense" in label else REGSMALL_MARKER
        size = 260 if marker == "*" else 130
        ax.scatter(
            [k], [v],
            color=color, marker=marker, s=size,
            edgecolor="black", linewidth=0.8, zorder=5,
            clip_on=False,
        )
        ax.annotate(
            f"{v:.1f}", xy=(k, v), xytext=(0, -14),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color=color, fontweight="bold",
        )
        all_vals.append(v)

    ax.set_xscale("log", base=2)
    ax.set_xticks(KEEPK_AXIS)
    ax.set_xticklabels([str(k) for k in KEEPK_AXIS])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.invert_xaxis()

    if all_vals:
        ymin_data = min(all_vals)
        ymax_data = max(all_vals)
        if random_chance is not None:
            ymin_data = min(ymin_data, random_chance)
        span = ymax_data - ymin_data
        pad = max(span * 0.12, 0.5)
        ax.set_ylim(bottom=ymin_data - pad, top=ymax_data + pad * 1.1)

    if random_chance is not None:
        ax.axhline(
            random_chance, color="#666666", linewidth=1.3,
            linestyle=(0, (4, 3)), zorder=2, alpha=0.8,
        )
        ax.text(
            ax.get_xlim()[1], random_chance, "  random",
            ha="left", va="center", fontsize=9, color="#666666",
            fontweight="bold", clip_on=False, zorder=3,
        )


def render_figure(df: pd.DataFrame, output_path: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _draw_panel(
        ax, df, metric_col, flex_model, reg_model, refs,
        random_chance=random_chance,
    )

    ax.set_title(f"MMLU after fine-tuning (130B)", pad=10, fontsize=13)
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
            label="Dense @8 (trained)",
        ),
        mlines.Line2D(
            [], [], color="none", marker=REGSMALL_MARKER, markersize=10,
            markerfacecolor=REGSMALL_COLOR,
            markeredgecolor="black", markeredgewidth=0.8,
            label="Reg. MoE @32 (trained)",
        ),
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        frameon=True, fontsize=9,
        handletextpad=0.5, borderpad=0.6,
        facecolor="white", edgecolor="#CCCCCC",
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(args.input)
    render_figure(df, args.output)


if __name__ == "__main__":
    main()
