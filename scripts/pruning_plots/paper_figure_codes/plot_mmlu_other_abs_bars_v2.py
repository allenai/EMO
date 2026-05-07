#!/usr/bin/env python3
"""Variant of plot_mmlu_other_abs_bars.py with legends placed at the TOP.

Same data, same panel layout, same color palettes — only the three legend
boxes (Reg. MoE shades, EMO shades, Trained baselines) are moved above
the panels instead of below.

Outputs:
    claude_outputs/prune_plots/mmlu_other_abs_bars_130b_v2.pdf
    claude_outputs/prune_plots/mmlu_other_abs_bars_1t_v2.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_mmlu_other_abs_bars import (
    DENSE_COLOR,
    DEFAULT_OUTPUT_DIR,
    FLEX_PALETTE,
    KEEPK_VALUES,
    PANEL_SOURCE,
    PANEL_TITLES,
    PANEL_Y_CONFIG,
    PRUNE_PLOTS_DIR,
    REG_PALETTE,
    REGSMALL_COLOR,
    SCALES,
    _build_clumps,
    _draw_panel,
    _load_panel_dfs,
    _resolve_finetune_dir,
    _shade_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inference-dir", type=Path, default=None)
    parser.add_argument("--finetune-dir", type=Path, default=None)
    parser.add_argument(
        "--scale", choices=["130b", "1t", "both"], default="both",
        help="Which training-scale figure to render. Default: both.",
    )
    return parser.parse_args()


def _draw_per_shade_legend_top(fig, *, with_refs: bool) -> None:
    """Three legend boxes at the TOP of the figure (mirrors the bottom-row
    layout used by the v1 script, but anchored at y near 1.0)."""
    keepk_labels = [_shade_label(k) for k in KEEPK_VALUES]
    common_kw = dict(
        frameon=True, fontsize=9.5, title_fontsize=11,
        handletextpad=0.4, columnspacing=0.8,
        handlelength=1.3, handleheight=1.0,
        borderpad=0.7, labelspacing=0.4,
    )

    if with_refs:
        reg_x, flex_x, refs_x = 0.27, 0.55, 0.83
    else:
        reg_x, flex_x, refs_x = 0.35, 0.65, None

    reg_handles = [
        mpatches.Patch(facecolor=REG_PALETTE[i], edgecolor="black", linewidth=0.6)
        for i in range(len(KEEPK_VALUES))
    ]
    leg_r = fig.legend(
        reg_handles, keepk_labels, title="Reg. MoE",
        ncol=len(reg_handles),
        loc="upper center", bbox_to_anchor=(reg_x, 0.995),
        facecolor="#EEF6E4", edgecolor=REG_PALETTE[1],
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
        flex_handles, keepk_labels, title="EMO",
        ncol=len(flex_handles),
        loc="upper center", bbox_to_anchor=(flex_x, 0.995),
        facecolor="#FBE7EF", edgecolor=FLEX_PALETTE[2],
        **common_kw,
    )
    leg_f.get_title().set_fontweight("bold")
    leg_f.get_title().set_color(FLEX_PALETTE[0])
    leg_f.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_f)

    if with_refs:
        ref_handles = [
            mpatches.Patch(facecolor="white", edgecolor=DENSE_COLOR,
                           hatch="///", linewidth=1.4),
            mpatches.Patch(facecolor="white", edgecolor=REGSMALL_COLOR,
                           hatch="///", linewidth=1.4),
        ]
        ref_labels = ["Dense @8 (trained)", "Reg. MoE @32 (trained)"]
        leg_x = fig.legend(
            ref_handles, ref_labels, title="Trained baselines",
            ncol=len(ref_handles),
            loc="upper center", bbox_to_anchor=(refs_x, 0.995),
            facecolor="#F5F5F5", edgecolor="#999999",
            **common_kw,
        )
        leg_x.get_title().set_fontweight("bold")
        leg_x.get_title().set_color("#444444")
        leg_x.get_frame().set_linewidth(1.0)
        fig.add_artist(leg_x)


def render_figure_top_legend(
    inf_dfs: Dict[str, Dict[str, pd.DataFrame]],
    *,
    scale: str,
    output_path: Path,
) -> None:
    spec = SCALES[scale]
    refs = spec["refs"]

    mode = "inf"
    row_label = "Inference\n(no fine-tune)"

    n_cols = len(PANEL_TITLES)
    fig, axes = plt.subplots(1, n_cols, figsize=(14, 4.6))
    handles_for_legend: Dict[str, plt.Artist] = {}

    for c, panel_title in enumerate(PANEL_TITLES):
        ax = axes[c]
        df = inf_dfs[mode][panel_title]
        metric_col = PANEL_SOURCE[panel_title][1]
        clumps, overlays = _build_clumps(
            df, metric_col,
            spec["reg_row_template"], spec["flex_row_template"], refs,
        )
        panel_cfg = PANEL_Y_CONFIG.get(panel_title, {})
        _draw_panel(ax, clumps, overlays, panel_cfg, handles_for_legend)
        # No per-panel title — legend rows live above the figure now.
        ax.set_title(panel_title)
        ax.set_xlabel("")
        ax.set_ylabel(f"{row_label}\nPerformance" if c == 0 else "")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.xaxis.grid(False)

    _draw_per_shade_legend_top(fig, with_refs=bool(refs))
    # Reserve the top ~16% of the figure for the three legend boxes.
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")

    inf_dir = (
        args.inference_dir
        if args.inference_dir is not None
        else PRUNE_PLOTS_DIR / "prune_eval_tables_final_ckpt0"
    )
    inf_dfs = {"inf": _load_panel_dfs(inf_dir / "acc_raw")}

    scales = list(SCALES.keys()) if args.scale == "both" else [args.scale]
    for scale in scales:
        render_figure_top_legend(
            inf_dfs,
            scale=scale,
            output_path=args.output_dir / f"mmlu_other_abs_bars_{scale}_v2.pdf",
        )


if __name__ == "__main__":
    main()
