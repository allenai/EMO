#!/usr/bin/env python3
"""Absolute-performance bar chart of ``main_results_table.csv``.

2x3 layout — top row = 130B, bottom row = 1T; columns = MMLU / MMLU Pro /
GSM8K. Within each panel, bars are grouped into per-model clumps so the
keepk-progression of each model reads left-to-right within its own clump:

    Clump 1 — Reg. MoE  (5 bars, keepk 128 -> 8, blue shades)
    Clump 2 — FlexMoE   (5 bars, keepk 128 -> 8, red shades)
    Clump 3 — Dense, trained @8                    (130B row only, gray)
    Clump 4 — Reg. MoE, trained @32 ("moe_small")  (130B row only, dark blue)

Within Reg. MoE / FlexMoE clumps, color saturation goes from dark (keepk
128) to light (keepk 8) so the "decay" is visible at a glance.

Outputs:
    claude_outputs/prune_plots/main_results_abs_bars_inf.png
    claude_outputs/prune_plots/main_results_abs_bars_ft.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "prune_plots"

MODES: Dict[str, Tuple[str, str]] = {
    "inf": ("inf", "Inference (no fine-tune)"),
    "ft": ("ft", "Fine-tune"),
}

# Per-scale spec drives one figure each.
# Filled in below once DENSE_COLOR / REGSMALL_COLOR are defined.
SCALES: Dict[str, Dict[str, object]] = {}

PANEL_TITLES = ["MMLU", "MMLU Pro", "GSM8K"]
KEEPK_VALUES = [128, 64, 32, 16, 8]  # left -> right within each clump

# Per-panel customization. Keys can be either a panel title (applies to both
# inf and ft modes) or the full metric column name like "GSM8K (ft)" (applies
# only to that specific mode of that task). Specific keys override generic
# ones. Supported entries:
#   - ymin: truncate y-axis bottom
#   - random_chance: dotted reference line at chance level
#   - overlay_text_overrides: {keepk: {**ax.annotate kwargs}} to nudge the
#     trained-baseline overlay annotation for one specific bar.
PANEL_Y_CONFIG: Dict[str, Dict[str, object]] = {
    "MMLU": {"ymin": 20.0, "random_chance": 25.0},      # 4-option MC -> 25%
    "MMLU Pro": {"ymin": 7.0, "random_chance": 10.0},   # 10-option MC -> 10%
    # GSM8K fine-tune values are tightly packed; nudge the Reg. MoE @32
    # trained overlay annotation upward so it doesn't collide with the
    # FlexMoE keepk=16 annotation to its right.
    "GSM8K (ft)": {
        "overlay_text_overrides": {
            32: {"xytext": (0, 24), "ha": "center", "va": "bottom"},
        },
    },
}

# Hand-picked gradient palettes inspired by an AI2-flavored aesthetic:
# - FlexMoE: deep purple -> magenta -> AI2-style pink (lightest end nods to
#   the AI2 brand pink while still sitting at the bottom of an ordered ramp).
# - Reg. MoE: dark forest green -> pale yellow-green.
# Indexed by keepk position (0 = darkest = keepk 128, 4 = lightest = keepk 8).
FLEX_PALETTE = ["#3F1052", "#6E1F73", "#B8327C", "#E48AB5", "#F2C7DC"]
REG_PALETTE = ["#225C2E", "#5B8E3F", "#93C265", "#C5DD93", "#E4EFC8"]

DENSE_COLOR = "#E78532"     # warm orange — Dense trained baseline overlay
REGSMALL_COLOR = "#1F6E7C"  # dark teal — Reg. MoE trained @32 baseline overlay

SCALES = {
    "130b": {
        "label": "130B",
        "reg_model": "Reg. MoE (130B)",
        "flex_model": "FlexMoE (130B)",
        "refs": [
            ("Dense, trained @8", "Dense (130B)^dagger", 8, DENSE_COLOR),
            ("Reg. MoE, trained @32", "Reg. MoE (130B)", 32, REGSMALL_COLOR),
        ],
    },
    "1t": {
        "label": "1T",
        "reg_model": "Reg. MoE (1T)",
        "flex_model": "FlexMoE (1T)",
        "refs": [],
    },
}

# Layout knobs.
BAR_WIDTH = 0.8
INTRA_BAR_STEP = 1.0  # x-distance between adjacent bars within a clump
CLUMP_GAP = 1.4       # extra x-gap between clumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scale", choices=["130b", "1t", "both"], default="both",
        help="Which training-scale figure to render. Default: both.",
    )
    return parser.parse_args()


def _parse_experts(value: str) -> int:
    m = re.match(r"\s*(\d+)", str(value))
    if m is None:
        raise ValueError(f"Cannot parse expert count from {value!r}")
    return int(m.group(1))


def _is_trained(value: str) -> bool:
    return "(trained)" in str(value)


def _lookup(
    df: pd.DataFrame, model_name: str, keepk: int, metric_col: str,
    *, prefer_trained: bool = False,
) -> Optional[float]:
    sub = df[df.iloc[:, 0] == model_name].copy()
    sub = sub[sub["# Total Experts"].map(_parse_experts) == keepk]
    if sub.empty:
        return None
    sub["trained_flag"] = sub["# Total Experts"].map(_is_trained).astype(int)
    sub = sub.sort_values(by="trained_flag", ascending=not prefer_trained)
    val = sub.iloc[0][metric_col]
    return None if pd.isna(val) else float(val)


def _shade(base_hex: str, weight: float) -> str:
    """Mix ``base_hex`` toward white. weight=1.0 -> pure base; lower -> lighter."""
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    mixed = white * (1 - weight) + base * weight
    return mcolors.to_hex(mixed)


def _gradient(base_hex: str, n: int, min_weight: float) -> List[str]:
    """Return n shades of base_hex from saturated (idx 0, weight 1.0) to
    a tinted version (idx n-1, weight=min_weight). Lower min_weight = wider
    spread of shades."""
    if n <= 1:
        return [_shade(base_hex, 1.0)]
    return [
        _shade(base_hex, 1.0 - (1.0 - min_weight) * (i / (n - 1)))
        for i in range(n)
    ]


def _build_clumps(
    df: pd.DataFrame,
    metric_col: str,
    reg_model: str,
    flex_model: str,
    refs: List[Tuple[str, str, int, str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build (clumps, overlays).

    clumps: ordered list of {legend_label, x_label, bars} where bars is a
    list of (sub_label, value, color, hatch). Always Reg. MoE then FlexMoE.

    overlays: list of {legend_label, target_clump_idx, target_keepk, value,
    color} — drawn on top of FlexMoE bars at the matching keepk position.
    """
    n = len(KEEPK_VALUES)
    if len(REG_PALETTE) < n or len(FLEX_PALETTE) < n:
        raise RuntimeError(
            f"Palettes must contain at least {n} colors; "
            f"got REG={len(REG_PALETTE)}, FLEX={len(FLEX_PALETTE)}"
        )

    reg_bars = []
    flex_bars = []
    for i, k in enumerate(KEEPK_VALUES):
        reg_bars.append((str(k), _lookup(df, reg_model, k, metric_col),
                         REG_PALETTE[i], ""))
        flex_bars.append((str(k), _lookup(df, flex_model, k, metric_col),
                          FLEX_PALETTE[i], ""))

    clumps: List[Dict[str, object]] = [
        {"legend_label": "Reg. MoE", "x_label": "Reg. MoE", "bars": reg_bars},
        {"legend_label": "FlexMoE", "x_label": "FlexMoE", "bars": flex_bars},
    ]

    overlays: List[Dict[str, object]] = []
    for label, model_name, k, color in refs:
        if k not in KEEPK_VALUES:
            continue
        v = _lookup(df, model_name, k, metric_col, prefer_trained=True)
        if v is None:
            continue
        overlays.append(
            {
                "legend_label": label,
                "target_clump_idx": 1,  # FlexMoE
                "target_keepk": k,
                "value": v,
                "color": color,
            }
        )

    return clumps, overlays


def _draw_panel(
    ax,
    clumps: List[Dict[str, object]],
    overlays: List[Dict[str, object]],
    panel_config: Dict[str, float],
    handles_for_legend: Dict[str, plt.Artist],
) -> None:
    cursor = 0.0
    clump_centers: List[float] = []
    clump_labels: List[str] = []
    sub_positions: List[float] = []
    sub_labels: List[str] = []
    all_vals: List[float] = []

    # (clump_idx, bar_idx) -> x position, used to place overlays.
    bar_x_by_key: Dict[Tuple[int, int], float] = {}

    for ci, clump in enumerate(clumps):
        bars = clump["bars"]
        n = len(bars)
        positions = [cursor + i * INTRA_BAR_STEP for i in range(n)]
        clump_start = positions[0]
        clump_end = positions[-1]
        clump_centers.append((clump_start + clump_end) / 2)
        clump_labels.append(clump["x_label"])

        xs = []
        ys = []
        colors = []
        hatches = []
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

        # Per-bar labels: numeric value above + keepk under.
        for (sub, val, _c, _h), x in zip(bars, positions):
            if val is not None:
                ax.annotate(
                    f"{val:.1f}", xy=(x, val), xytext=(0, 2),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color="#222222",
                )
            sub_positions.append(x)
            sub_labels.append(sub)

        cursor = clump_end + INTRA_BAR_STEP + CLUMP_GAP

    # Overlays: drawn after all clumps so they sit on top. Style as a thin
    # horizontal cap (a wide, very-low rectangle) at the baseline value, plus
    # a vertical line up from x-axis colored to match — gives a "trained
    # baseline reaches THIS height" reading without obscuring the FlexMoE bar.
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
        # Translucent vertical fill from 0 -> baseline value.
        ax.bar(
            x, v, BAR_WIDTH * 1.02,
            color="white", edgecolor=color, linewidth=1.6,
            hatch="///", alpha=0.6, zorder=4,
            label=ov["legend_label"],
        )
        # Solid colored cap line emphasising the baseline level.
        ax.plot(
            [x - BAR_WIDTH / 2, x + BAR_WIDTH / 2],
            [v, v],
            color=color, linewidth=2.6, solid_capstyle="butt", zorder=5,
        )
        annotate_kwargs = {
            "xytext": (BAR_WIDTH * 14, 0),
            "ha": "left", "va": "center",
        }
        annotate_kwargs.update(overlay_overrides.get(ov["target_keepk"], {}))
        ax.annotate(
            f"{v:.1f}", xy=(x, v),
            textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold", zorder=6,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                            shrinkA=0, shrinkB=2),
            **annotate_kwargs,
        )
        # Track legend handle (use a representative patch).
        if ov["legend_label"] not in handles_for_legend:
            handles_for_legend[ov["legend_label"]] = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="white", edgecolor=color, hatch="///", linewidth=1.6,
                label=ov["legend_label"],
            )

    # Bar identification is conveyed by the legend, so drop x-tick keepk
    # labels and the clump model names entirely.
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


def _shade_label(k: int) -> str:
    return f"{k} (full)" if k == 128 else str(k)


def _draw_per_shade_legend(fig, *, with_refs: bool) -> None:
    """Three centered legend boxes at the bottom:

        [ Reg. MoE  shades ]   [ FlexMoE shades ]   [ Trained baselines ]

    Reg. MoE box has a soft green wash; FlexMoE box has a soft pink wash;
    trained baselines (Dense, Reg. MoE @32) live in their own neutral box.
    """
    keepk_labels = [_shade_label(k) for k in KEEPK_VALUES]
    common_kw = dict(
        frameon=True,
        fontsize=9.5,
        title_fontsize=11,
        handletextpad=0.4,
        columnspacing=0.8,
        handlelength=1.3,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
    )

    # Centers shift depending on whether the trained-baselines box is shown,
    # so the visible group of legend boxes stays roughly centered.
    if with_refs:
        reg_x, flex_x, refs_x = 0.27, 0.55, 0.83
    else:
        reg_x, flex_x, refs_x = 0.35, 0.65, None

    # ---- Reg. MoE box ----
    reg_handles = [
        mpatches.Patch(facecolor=REG_PALETTE[i], edgecolor="black", linewidth=0.6)
        for i in range(len(KEEPK_VALUES))
    ]
    leg_r = fig.legend(
        reg_handles, keepk_labels,
        title="Reg. MoE",
        ncol=len(reg_handles),
        loc="lower center", bbox_to_anchor=(reg_x, 0.005),
        facecolor="#EEF6E4", edgecolor=REG_PALETTE[1],
        **common_kw,
    )
    leg_r.get_title().set_fontweight("bold")
    leg_r.get_title().set_color(REG_PALETTE[0])
    leg_r.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_r)

    # ---- FlexMoE box ----
    flex_handles = [
        mpatches.Patch(facecolor=FLEX_PALETTE[i], edgecolor="black", linewidth=0.6)
        for i in range(len(KEEPK_VALUES))
    ]
    leg_f = fig.legend(
        flex_handles, keepk_labels,
        title="ModMoE",
        ncol=len(flex_handles),
        loc="lower center", bbox_to_anchor=(flex_x, 0.005),
        facecolor="#FBE7EF", edgecolor=FLEX_PALETTE[2],
        **common_kw,
    )
    leg_f.get_title().set_fontweight("bold")
    leg_f.get_title().set_color(FLEX_PALETTE[0])
    leg_f.get_frame().set_linewidth(1.1)
    fig.add_artist(leg_f)

    # ---- Trained baselines box (only when overlays are present) ----
    if with_refs:
        ref_handles = [
            mpatches.Patch(facecolor="white", edgecolor=DENSE_COLOR,
                           hatch="///", linewidth=1.4),
            mpatches.Patch(facecolor="white", edgecolor=REGSMALL_COLOR,
                           hatch="///", linewidth=1.4),
        ]
        ref_labels = ["Dense @8 (trained)", "Reg. MoE @32 (trained)"]
        leg_x = fig.legend(
            ref_handles, ref_labels,
            title="Trained baselines",
            ncol=len(ref_handles),
            loc="lower center", bbox_to_anchor=(refs_x, 0.005),
            facecolor="#F5F5F5", edgecolor="#999999",
            **common_kw,
        )
        leg_x.get_title().set_fontweight("bold")
        leg_x.get_title().set_color("#444444")
        leg_x.get_frame().set_linewidth(1.0)
        fig.add_artist(leg_x)


def render_figure(
    df: pd.DataFrame,
    *,
    scale: str,
    output_path: Path,
) -> None:
    spec = SCALES[scale]
    reg_model = spec["reg_model"]
    flex_model = spec["flex_model"]
    refs = spec["refs"]

    # Rows: top = inference, bottom = fine-tune.
    row_specs = [
        ("inf", "Inference\n(no fine-tune)"),
        ("ft", "Fine-tune"),
    ]

    n_rows = len(row_specs)
    n_cols = len(PANEL_TITLES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 3.4 * n_rows + 1.2)
    )
    handles_for_legend: Dict[str, plt.Artist] = {}

    for r, (suffix, row_label) in enumerate(row_specs):
        for c, panel_title in enumerate(PANEL_TITLES):
            ax = axes[r][c]
            metric_col = f"{panel_title} ({suffix})"
            clumps, overlays = _build_clumps(
                df, metric_col, reg_model, flex_model, refs
            )
            panel_cfg = {
                **PANEL_Y_CONFIG.get(panel_title, {}),
                **PANEL_Y_CONFIG.get(metric_col, {}),
            }
            _draw_panel(ax, clumps, overlays, panel_cfg, handles_for_legend)
            if r == 0:
                ax.set_title(panel_title)
            ax.set_xlabel("")
            ax.set_ylabel(f"{row_label}\nPerformance" if c == 0 else "")
            ax.tick_params(axis="x", length=0, pad=2)
            ax.xaxis.grid(False)

    _draw_per_shade_legend(fig, with_refs=bool(refs))

    fig.tight_layout(rect=(0, 0.09, 1, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.read_csv(args.input)
    scales = list(SCALES.keys()) if args.scale == "both" else [args.scale]
    for scale in scales:
        render_figure(
            df,
            scale=scale,
            output_path=args.output_dir / f"main_results_abs_bars_{scale}.png",
        )


if __name__ == "__main__":
    main()
