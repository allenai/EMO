#!/usr/bin/env python3
"""Absolute-performance bar chart for the *other* subjects only.

Sibling of ``plot_main_results_abs_bars.py``, but focused on the two
subjects that the main figure deliberately excludes from its averages:

    - ``mmlu_merged_other``     (the "other" MMLU bucket)
    - ``mmlu_pro_merged_other`` (the "other" MMLU Pro bucket)

Same 2x(N) layout as the main figure — top row = inference, bottom row =
fine-tune — but with TWO columns instead of three (no GSM8K analog
exists for "other"). Per-model bar clumps, per-keepk shaded bars, and
the same trained-baseline overlays on FlexMoE bars in the 130B figure.

Reads the per-subject CSVs produced by
``get_table_scores_prune_evals_final.py``:

    inference: claude_outputs/prune_plots/prune_eval_tables_final_ckpt0/acc_raw/
                   {mmlu_merged.csv, mmlu_pro_merged.csv}
    fine-tune: claude_outputs/prune_plots/prune_eval_tables_final/acc_raw/
                   {mmlu_merged.csv, mmlu_pro_merged.csv}
                   (falls back to the most recent
                   prune_eval_tables_final_*backup* directory if the
                   plain dir doesn't exist).

Outputs:
    claude_outputs/prune_plots/mmlu_other_abs_bars_130b.pdf
    claude_outputs/prune_plots/mmlu_other_abs_bars_1t.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots"
PRUNE_PLOTS_DIR = DEFAULT_OUTPUT_DIR

PANEL_TITLES = ["MMLU other", "MMLU Pro other"]
KEEPK_VALUES = [128, 64, 32, 16, 8]  # left -> right within each clump

# Per-panel customization. Same chance levels as MMLU / MMLU Pro because the
# "other" subjects share the same option-count format.
PANEL_Y_CONFIG: Dict[str, Dict[str, object]] = {
    "MMLU other": {"ymin": 20.0, "random_chance": 25.0},
    "MMLU Pro other": {"ymin": 7.0, "random_chance": 10.0},
}

# Same palettes as plot_main_results_abs_bars.py.
FLEX_PALETTE = ["#3F1052", "#6E1F73", "#B8327C", "#E48AB5", "#F2C7DC"]
REG_PALETTE = ["#225C2E", "#5B8E3F", "#93C265", "#C5DD93", "#E4EFC8"]
DENSE_COLOR = "#E78532"
REGSMALL_COLOR = "#1F6E7C"

# Per-panel: which CSV file under acc_raw/ holds it, and which "(lw)" column
# represents the "other" subject's accuracy.
PANEL_SOURCE: Dict[str, Tuple[str, str]] = {
    "MMLU other":     ("mmlu_merged.csv",     "mmlu_merged_other (lw)"),
    "MMLU Pro other": ("mmlu_pro_merged.csv", "mmlu_pro_merged_other (lw)"),
}

# Per-scale spec: how to look up each model in the row-index of the per-subject
# CSV (model rows use the source naming convention,
# e.g. "moe (keepk 8)", "moe 1T + anneal (keepk 32)", "dense", "moe_small").
SCALES: Dict[str, Dict[str, object]] = {
    "130b": {
        "label": "130B",
        "reg_row_template": "moe (keepk {k})",
        "flex_row_template": (
            "specialized moe + globallb + 1shardexp + randpool (keepk {k})"
        ),
        "refs": [
            # (display label, exact CSV row name, target keepk on FlexMoE, color)
            ("Dense @8 (trained)", "dense", 8, DENSE_COLOR),
            ("Reg. MoE @32 (trained)", "moe_small", 32, REGSMALL_COLOR),
        ],
    },
    "1t": {
        "label": "1T",
        "reg_row_template": "moe 1T + anneal (keepk {k})",
        "flex_row_template": "specialized moe 1T + anneal (keepk {k})",
        "refs": [],
    },
}

BAR_WIDTH = 0.8
INTRA_BAR_STEP = 1.0
CLUMP_GAP = 1.4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--inference-dir", type=Path, default=None,
        help="Path to prune_eval_tables_final_ckpt0/. Default: under output-dir.",
    )
    parser.add_argument(
        "--finetune-dir", type=Path, default=None,
        help=(
            "Path to prune_eval_tables_final/. Defaults to the canonical "
            "location with a fallback to the most recent "
            "prune_eval_tables_final_*backup* directory."
        ),
    )
    parser.add_argument(
        "--scale", choices=["130b", "1t", "both"], default="both",
        help="Which training-scale figure to render. Default: both.",
    )
    return parser.parse_args()


def _resolve_finetune_dir(arg_dir: Optional[Path]) -> Path:
    if arg_dir is not None:
        if not arg_dir.is_dir():
            raise FileNotFoundError(f"--finetune-dir does not exist: {arg_dir}")
        return arg_dir
    primary = PRUNE_PLOTS_DIR / "prune_eval_tables_final"
    if (primary / "acc_raw" / "mmlu_merged.csv").is_file():
        return primary
    backups = sorted(
        [p for p in PRUNE_PLOTS_DIR.glob("prune_eval_tables_final_*backup*") if p.is_dir()],
        reverse=True,
    )
    for c in backups:
        if (c / "acc_raw" / "mmlu_merged.csv").is_file():
            print(f"[INFO] Using backup fine-tuning dir: {c}")
            return c
    raise FileNotFoundError(
        f"No fine-tune table dir found at {primary} or in *backup* siblings."
    )


# ---------------------------------------------------------------------------
# Data + color helpers (mirror plot_main_results_abs_bars.py)
# ---------------------------------------------------------------------------


def _shade(base_hex: str, weight: float) -> str:
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    return mcolors.to_hex(white * (1 - weight) + base * weight)


def _lookup(df: pd.DataFrame, row_name: str, col_name: str) -> Optional[float]:
    if row_name not in df.index or col_name not in df.columns:
        return None
    val = df.loc[row_name, col_name]
    if pd.isna(val):
        return None
    return float(val) * 100.0  # CSV stores 0-1 fractions; figure displays %


def _build_clumps(
    panel_df: pd.DataFrame,
    metric_col: str,
    reg_template: str,
    flex_template: str,
    refs: List[Tuple[str, str, int, str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n = len(KEEPK_VALUES)
    if len(REG_PALETTE) < n or len(FLEX_PALETTE) < n:
        raise RuntimeError(
            f"Palettes must have >= {n} colors; "
            f"got REG={len(REG_PALETTE)}, FLEX={len(FLEX_PALETTE)}"
        )
    reg_bars = []
    flex_bars = []
    for i, k in enumerate(KEEPK_VALUES):
        reg_bars.append((
            str(k),
            _lookup(panel_df, reg_template.format(k=k), metric_col),
            REG_PALETTE[i], "",
        ))
        flex_bars.append((
            str(k),
            _lookup(panel_df, flex_template.format(k=k), metric_col),
            FLEX_PALETTE[i], "",
        ))

    clumps: List[Dict[str, object]] = [
        {"legend_label": "Reg. MoE", "x_label": "Reg. MoE", "bars": reg_bars},
        {"legend_label": "EMO",   "x_label": "EMO",   "bars": flex_bars},
    ]

    overlays: List[Dict[str, object]] = []
    for label, row_name, k, color in refs:
        if k not in KEEPK_VALUES:
            continue
        v = _lookup(panel_df, row_name, metric_col)
        if v is None:
            continue
        overlays.append({
            "legend_label": label,
            "target_clump_idx": 1,  # FlexMoE
            "target_keepk": k,
            "value": v,
            "color": color,
        })
    return clumps, overlays


def _draw_panel(
    ax,
    clumps: List[Dict[str, object]],
    overlays: List[Dict[str, object]],
    panel_config: Dict[str, object],
    handles_for_legend: Dict[str, plt.Artist],
) -> None:
    cursor = 0.0
    all_vals: List[float] = []
    bar_x_by_key: Dict[Tuple[int, int], float] = {}

    for ci, clump in enumerate(clumps):
        bars = clump["bars"]
        positions = [cursor + i * INTRA_BAR_STEP for i in range(len(bars))]
        xs, ys, colors, hatches = [], [], [], []
        for bi, ((_sub, val, color, hatch), x) in enumerate(zip(bars, positions)):
            xs.append(x)
            ys.append(val if val is not None else 0.0)
            colors.append(color)
            hatches.append(hatch)
            bar_x_by_key[(ci, bi)] = x
            if val is not None:
                all_vals.append(val)

        bc = ax.bar(
            xs, ys, BAR_WIDTH,
            color=colors, edgecolor="black", linewidth=0.6,
            label=clump["legend_label"],
        )
        for rect, hatch in zip(bc, hatches):
            if hatch:
                rect.set_hatch(hatch)
        handles_for_legend.setdefault(clump["legend_label"], bc)

        for (sub, val, _c, _h), x in zip(bars, positions):
            if val is not None:
                ax.annotate(
                    f"{val:.1f}", xy=(x, val), xytext=(0, 2),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color="#222222",
                )

        cursor = positions[-1] + INTRA_BAR_STEP + CLUMP_GAP

    # Trained-baseline overlays on FlexMoE bars.
    overlay_overrides = panel_config.get("overlay_text_overrides", {})
    for ov in overlays:
        ci = ov["target_clump_idx"]
        if ov["target_keepk"] not in KEEPK_VALUES:
            continue
        bi = KEEPK_VALUES.index(ov["target_keepk"])
        x = bar_x_by_key.get((ci, bi))
        if x is None:
            continue
        v = ov["value"]
        color = ov["color"]
        ax.bar(
            x, v, BAR_WIDTH * 1.02,
            color="white", edgecolor=color, linewidth=1.6,
            hatch="///", alpha=0.6, zorder=4,
            label=ov["legend_label"],
        )
        ax.plot(
            [x - BAR_WIDTH / 2, x + BAR_WIDTH / 2], [v, v],
            color=color, linewidth=2.6, solid_capstyle="butt", zorder=5,
        )
        annotate_kwargs = {
            "xytext": (BAR_WIDTH * 14, 0),
            "ha": "left", "va": "center",
        }
        annotate_kwargs.update(overlay_overrides.get(ov["target_keepk"], {}))
        ax.annotate(
            f"{v:.1f}", xy=(x, v), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold", zorder=6,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                            shrinkA=0, shrinkB=2),
            **annotate_kwargs,
        )
        if ov["legend_label"] not in handles_for_legend:
            handles_for_legend[ov["legend_label"]] = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="white", edgecolor=color, hatch="///", linewidth=1.6,
                label=ov["legend_label"],
            )

    ax.set_xticks([])
    ymin = float(panel_config.get("ymin", 0.0))
    if all_vals:
        ymax = max(all_vals)
        top = ymin + (ymax - ymin) * 1.18 if ymax > ymin else ymax * 1.18
        ax.set_ylim(top=top, bottom=ymin)

    chance = panel_config.get("random_chance")
    if chance is not None:
        ax.axhline(
            chance, color="#222222", linewidth=1.6,
            linestyle=(0, (4, 3)), zorder=6, alpha=0.85,
        )
        x_right = ax.get_xlim()[1]
        ax.text(
            x_right, chance, "  random",
            ha="left", va="center", fontsize=9, color="#222222",
            fontweight="bold", clip_on=False, zorder=6,
        )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def _shade_label(k: int) -> str:
    return f"{k} (full)" if k == 128 else str(k)


def _draw_per_shade_legend(fig, *, with_refs: bool) -> None:
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
        loc="lower center", bbox_to_anchor=(reg_x, 0.005),
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
        loc="lower center", bbox_to_anchor=(flex_x, 0.005),
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
            loc="lower center", bbox_to_anchor=(refs_x, 0.005),
            facecolor="#F5F5F5", edgecolor="#999999",
            **common_kw,
        )
        leg_x.get_title().set_fontweight("bold")
        leg_x.get_title().set_color("#444444")
        leg_x.get_frame().set_linewidth(1.0)
        fig.add_artist(leg_x)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _load_panel_dfs(acc_raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Returns a dict {panel_title -> per-subject CSV indexed by model name}."""
    out = {}
    for panel_title, (csv_name, _col) in PANEL_SOURCE.items():
        path = acc_raw_dir / csv_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing per-subject CSV: {path}")
        df = pd.read_csv(path).set_index("model")
        out[panel_title] = df
    return out


def render_figure(
    inf_dfs: Dict[str, Dict[str, pd.DataFrame]],
    *,
    scale: str,
    output_path: Path,
) -> None:
    spec = SCALES[scale]
    refs = spec["refs"]

    # Single row: inference only (no fine-tune).
    mode = "inf"
    row_label = "Inference\n(no fine-tune)"

    n_cols = len(PANEL_TITLES)
    # Match the figure width used by plot_main_results_abs_bars.py so the
    # legend boxes (which are sized by content, not figure width) sit with
    # the same horizontal proportions as the main figure.
    fig, axes = plt.subplots(
        1, n_cols, figsize=(14, 4.6),
    )
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
        ax.set_title(panel_title)
        ax.set_xlabel("")
        ax.set_ylabel(f"{row_label}\nPerformance" if c == 0 else "")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.xaxis.grid(False)

    _draw_per_shade_legend(fig, with_refs=bool(refs))
    fig.tight_layout(rect=(0, 0.16, 1, 1.0))

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
    # Inference-only figure — fine-tune dir not needed.
    inf_dfs = {"inf": _load_panel_dfs(inf_dir / "acc_raw")}

    scales = list(SCALES.keys()) if args.scale == "both" else [args.scale]
    for scale in scales:
        render_figure(
            inf_dfs,
            scale=scale,
            output_path=args.output_dir / f"mmlu_other_abs_bars_{scale}.pdf",
        )


if __name__ == "__main__":
    main()
