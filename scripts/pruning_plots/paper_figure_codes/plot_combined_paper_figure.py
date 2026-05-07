#!/usr/bin/env python3
"""Combined 3-panel paper figure.

Layout (1x3):
    Left   — 130B Pareto plot (EMO vs Reg. MoE pruning curves on MMLU-ft,
             with Dense @8 and Reg. MoE @32 trained baselines overlaid).
             Reuses ``plot_main_results_abs_bars_130b_shrunken``.
    Middle — Few-shot importance on MMLU, restricted to the "few-shot
             select + few-shot eval" configuration. Reuses
             ``plot_validation_fewshot_importance_clustered``.
    Right  — Expert-selection method on GSM8K (Reg. MoE x EMO crossed with
             Random / Router / Easy-EP). Reuses
             ``plot_expert_selection_method_combined``.

Output:
    claude_outputs/prune_plots/combined_paper_figure.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plot_validation_fewshot_importance_clustered as vfic
import plot_expert_selection_method_combined as esmc
from plot_main_results_abs_bars_130b_shrunken import (
    DENSE_MARKER,
    FLEX_LINE_COLOR,
    FLEX_MARKER,
    REG_LINE_COLOR,
    REG_MARKER,
    REGSMALL_MARKER,
    _draw_panel as _draw_pareto_panel,
)
from plot_main_results_abs_bars import (
    DENSE_COLOR,
    PANEL_Y_CONFIG,
    REGSMALL_COLOR,
    SCALES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "claude_outputs" / "prune_plots" / "combined_paper_figure.pdf"

PARETO_INPUT = REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"
FEWSHOT_INPUT = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "validation_sample_ablation_ckpt0.csv"
)
ESMC_INPUT = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "expert_selection_method_ckpt0.csv"
)

PARETO_TASK = "MMLU"
PARETO_MODE_SUFFIX = "ft"

FEWSHOT_TASK = "MMLU"
FEWSHOT_CONFIG = vfic.CONFIGS[0]  # ("few-shot select + few-shot eval", "Router", "#B8327C")

ESMC_TASK = "GSM8K"

# Drop the "Random" expert-selection method from the right-panel render.
esmc.METHODS = [m for m in esmc.METHODS if m[0] != "Random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _draw_pareto(ax, df: pd.DataFrame) -> None:
    spec = SCALES["130b"]
    flex_model = spec["flex_model"]
    reg_model = spec["reg_model"]
    refs = spec["refs"]

    metric_col = f"{PARETO_TASK} ({PARETO_MODE_SUFFIX})"
    panel_cfg = {
        **PANEL_Y_CONFIG.get(PARETO_TASK, {}),
        **PANEL_Y_CONFIG.get(metric_col, {}),
    }
    random_chance = panel_cfg.get("random_chance")

    _draw_pareto_panel(
        ax, df, metric_col, flex_model, reg_model, refs,
        random_chance=random_chance,
    )

    ax.set_title("MMLU after fine-tuning (130B)", pad=10, fontsize=12)
    ax.set_xlabel("Expert subset size (memory budget)", fontsize=10)
    ax.set_ylabel("MMLU accuracy", fontsize=10)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        mlines.Line2D(
            [], [], color=FLEX_LINE_COLOR, linewidth=2.4,
            marker=FLEX_MARKER, markersize=8,
            markerfacecolor=FLEX_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.0,
            label="EMO",
        ),
        mlines.Line2D(
            [], [], color=REG_LINE_COLOR, linewidth=2.2,
            marker=REG_MARKER, markersize=7,
            markerfacecolor=REG_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Reg. MoE",
        ),
        mlines.Line2D(
            [], [], color="none", marker=DENSE_MARKER, markersize=12,
            markerfacecolor=DENSE_COLOR,
            markeredgecolor="black", markeredgewidth=0.7,
            label="Dense @8 (trained)",
        ),
        mlines.Line2D(
            [], [], color="none", marker=REGSMALL_MARKER, markersize=9,
            markerfacecolor=REGSMALL_COLOR,
            markeredgecolor="black", markeredgewidth=0.7,
            label="Reg. MoE @32 (trained)",
        ),
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        frameon=True, fontsize=8.5,
        handletextpad=0.5, borderpad=0.5,
        facecolor="white", edgecolor="#CCCCCC",
    )
    leg.get_frame().set_linewidth(0.8)


def _draw_fewshot(ax, df: pd.DataFrame) -> None:
    """Few-shot importance for one task + one config (n-spread within keepk bins)."""
    keepk_bins = vfic.KEEPK_VALUES
    n_levels = vfic.N_LEVELS
    bin_centers = list(range(len(keepk_bins)))
    n_offsets = vfic._n_offsets(len(n_levels))

    cfg_label, prunemode, color = FEWSHOT_CONFIG
    all_ys: List[float] = []

    for n_idx, (_n_label, n_token, marker) in enumerate(n_levels):
        xs: List[float] = []
        ys: List[float] = []
        for b_idx, keepk in enumerate(keepk_bins):
            v = vfic._value(df, prunemode, FEWSHOT_TASK, keepk, n_token)
            if v is None:
                continue
            xs.append(bin_centers[b_idx] + n_offsets[n_idx])
            ys.append(v)
            all_ys.append(v)
        if xs:
            ax.scatter(
                xs, ys, color=color, marker=marker, s=80,
                edgecolor="black", linewidth=0.6, zorder=3,
                clip_on=False,
            )

    for b_idx in range(len(keepk_bins) - 1):
        ax.axvline(
            b_idx + 0.5, color="#cccccc", linewidth=0.8,
            linestyle="-", zorder=1,
        )

    ax.set_xticks(bin_centers)
    ax.set_xticklabels([str(k) for k in keepk_bins])
    ax.set_xlim(-0.5, len(keepk_bins) - 0.5)

    panel_cfg = vfic.PANEL_Y_CONFIG.get(FEWSHOT_TASK, {})
    ymin_cfg = float(panel_cfg.get("ymin", 0.0))
    random_chance = 25.0  # MMLU = 4-option MC
    if all_ys:
        y_min_data = min(all_ys)
        y_range = max(max(all_ys) - y_min_data, 1.0)
        pad = y_range * 0.10
        effective_ymin = min(ymin_cfg, y_min_data - pad, random_chance - pad)
        ax.set_ylim(bottom=effective_ymin, top=47.0)

    ax.axhline(
        random_chance, color="#666666", linewidth=1.3,
        linestyle=(0, (4, 3)), zorder=2, alpha=0.8,
    )
    ax.text(
        ax.get_xlim()[1], random_chance, "  random",
        ha="left", va="center", fontsize=8.5, color="#666666",
        fontweight="bold", clip_on=False, zorder=3,
    )

    ax.set_title(f"{FEWSHOT_TASK} — validation sample size", pad=10, fontsize=12)
    ax.set_xlabel("Expert subset size", fontsize=10)
    ax.set_ylabel("MMLU accuracy", fontsize=10)
    ax.grid(True, axis="y", alpha=0.5)
    ax.grid(False, axis="x")

    n_handles = [
        mlines.Line2D(
            [], [], color=color, marker=marker, linestyle="None", markersize=8,
            markeredgecolor="black", markeredgewidth=0.5, label=lbl,
        )
        for lbl, _tok, marker in n_levels
    ]
    leg = ax.legend(
        handles=n_handles, loc="upper center",
        title="n (validation samples)",
        frameon=True, fontsize=8, title_fontsize=8.5,
        ncol=len(n_handles),
        handletextpad=0.3, columnspacing=0.7, borderpad=0.4,
        facecolor="white", edgecolor="#CCCCCC",
    )
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_linewidth(0.7)


def _draw_esmc(ax, df: pd.DataFrame) -> None:
    panel_cfg = esmc.PANEL_Y_CONFIG.get(ESMC_TASK, {})
    esmc._draw_panel(ax, df, ESMC_TASK, panel_cfg)
    ax.set_title(f"{ESMC_TASK} — expert selection", pad=10, fontsize=12)
    ax.set_xlabel("Expert subset size", fontsize=10)
    ax.set_ylabel(f"{ESMC_TASK} accuracy", fontsize=10)
    ax.grid(True, axis="y", alpha=0.5)
    ax.grid(False, axis="x")

    cur_bottom, _ = ax.get_ylim()
    ax.set_ylim(bottom=cur_bottom, top=18.0)

    handles = []
    for model_label, _model_name, color in esmc.MODELS:
        for method_label, _suffix, linestyle, marker in esmc.METHODS:
            handles.append(
                mlines.Line2D(
                    [], [], color=color, linestyle=linestyle, marker=marker,
                    linewidth=2.0, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"{model_label} — {method_label}",
                )
            )
    leg = ax.legend(
        handles=handles, loc="upper right",
        frameon=True, fontsize=8, ncol=2,
        handletextpad=0.5, columnspacing=0.8, borderpad=0.5,
        facecolor="white", edgecolor="#CCCCCC",
    )
    leg.get_frame().set_linewidth(0.8)


def render_combined(output_path: Path) -> None:
    pareto_df = pd.read_csv(PARETO_INPUT)
    fewshot_df = pd.read_csv(FEWSHOT_INPUT)
    esmc_df = pd.read_csv(ESMC_INPUT)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    _draw_pareto(axes[0], pareto_df)
    _draw_fewshot(axes[1], fewshot_df)
    _draw_esmc(axes[2], esmc_df)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="notebook")
    render_combined(args.output)


if __name__ == "__main__":
    main()
