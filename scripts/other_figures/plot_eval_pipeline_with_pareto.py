#!/usr/bin/env python3
"""Combined figure: eval_pipeline.pdf (left, vector) + 130B Pareto plot (right).

Both panels stay vector. The left source is the existing
``other_figures/eval_pipeline.pdf`` (HTML-derived, vector via Chrome
printToPDF). The right is rendered by matplotlib to an intermediate
vector PDF, then both are overlaid on a single new page using pypdf.

Output:
    claude_outputs/other_figures/eval_pipeline_with_pareto.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject, NameObject

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = REPO_ROOT / "scripts" / "pruning_plots" / "paper_figure_codes"
sys.path.insert(0, str(PAPER_FIG_DIR))

from plot_main_results_abs_bars_130b_shrunken import (  # noqa: E402
    DENSE_MARKER,
    FLEX_LINE_COLOR,
    FLEX_MARKER,
    REG_LINE_COLOR,
    REG_MARKER,
    REGSMALL_MARKER,
    _draw_panel as _draw_pareto_panel,
)
from plot_main_results_abs_bars import (  # noqa: E402
    DENSE_COLOR,
    PANEL_Y_CONFIG,
    REGSMALL_COLOR,
    SCALES,
)

DEFAULT_PIPELINE_PDF = (
    REPO_ROOT / "claude_outputs" / "other_figures" / "eval_pipeline.pdf"
)
DEFAULT_PARETO_CSV = (
    REPO_ROOT / "claude_outputs" / "prune_plots" / "main_results_table.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "claude_outputs" / "other_figures" / "eval_pipeline_with_pareto.pdf"
)

PARETO_TASK = "MMLU"
PARETO_MODE_SUFFIX = "ft"

PARETO_WIDTH_IN = 5.8
PARETO_HEIGHT_IN = 4.6  # natural Pareto-plot height; if eval is taller, Pareto is centered.
GAP_IN = 0.55  # horizontal whitespace between the eval pipeline and the plot
PT_PER_IN = 72.0

# Match the eval_pipeline figure's typography.
INK_COLOR = "#1a1a1a"           # primary text color in the SVG diagram
MUTED_COLOR = "#7a7568"         # subtitle/muted color in the SVG diagram
PARETO_FONT_FAMILY = "EB Garamond"

# Fonts are bundled (as woff2) inside eval_pipeline.html. We unpack them
# the first time this script runs.
_FONT_DIR = REPO_ROOT / "claude_outputs" / "other_figures" / "_fonts"

# Default symmetric crop applied to eval_pipeline.pdf to trim edge whitespace.
EVAL_CROP_LEFT_IN = 0.0
EVAL_CROP_RIGHT_IN = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-pdf", type=Path, default=DEFAULT_PIPELINE_PDF)
    parser.add_argument("--pareto-csv", type=Path, default=DEFAULT_PARETO_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pareto-width-in", type=float, default=PARETO_WIDTH_IN)
    parser.add_argument("--pareto-height-in", type=float, default=PARETO_HEIGHT_IN)
    parser.add_argument("--gap-in", type=float, default=GAP_IN)
    parser.add_argument("--eval-crop-left-in", type=float, default=EVAL_CROP_LEFT_IN,
                        help="Trim this many inches off the LEFT edge of eval_pipeline.")
    parser.add_argument("--eval-crop-right-in", type=float, default=EVAL_CROP_RIGHT_IN,
                        help="Trim this many inches off the RIGHT edge of eval_pipeline.")
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

    ax.set_xlabel("Expert subset size (memory budget)", fontsize=27, color=INK_COLOR)
    ax.set_ylabel("MMLU accuracy", fontsize=27, color=INK_COLOR)
    ax.tick_params(axis="both", labelsize=21, colors=INK_COLOR)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(INK_COLOR)

    # Bump the value-label annotations baked into _draw_pareto_panel,
    # and reposition the "random" reference label so it sits above the
    # dotted line inside the plot rather than poking off to the right.
    for txt in ax.texts:
        if txt.get_text().strip() == "random":
            txt.set_text("random")
            txt.set_position((8, 25.6))
            txt.set_ha("right")
            txt.set_va("bottom")
            txt.set_fontsize(20)
            txt.set_color(MUTED_COLOR)
            txt.set_clip_on(True)
        elif txt.get_fontweight() in ("bold", 700):
            txt.set_fontsize(25)
        else:
            txt.set_fontsize(21)

    # Re-draw lines/markers from the imported _draw_pareto_panel at larger
    # weights so they don't look spindly on the bigger canvas.
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * 1.9)
        ms = line.get_markersize()
        if ms:
            line.set_markersize(ms * 1.9)
    for coll in ax.collections:
        sizes = coll.get_sizes()
        if len(sizes):
            coll.set_sizes(sizes * 2.4)

    handles = [
        mlines.Line2D(
            [], [], color=FLEX_LINE_COLOR, linewidth=4.2,
            marker=FLEX_MARKER, markersize=18,
            markerfacecolor=FLEX_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.8,
            label="EMO",
        ),
        mlines.Line2D(
            [], [], color=REG_LINE_COLOR, linewidth=4.0,
            marker=REG_MARKER, markersize=17,
            markerfacecolor=REG_LINE_COLOR,
            markeredgecolor="white", markeredgewidth=1.6,
            label="Reg. MoE",
        ),
        mlines.Line2D(
            [], [], color="none", marker=DENSE_MARKER, markersize=28,
            markerfacecolor=DENSE_COLOR,
            markeredgecolor="black", markeredgewidth=1.3,
            label="Dense @8 (trained)",
        ),
        mlines.Line2D(
            [], [], color="none", marker=REGSMALL_MARKER, markersize=21,
            markerfacecolor=REGSMALL_COLOR,
            markeredgecolor="black", markeredgewidth=1.3,
            label="Reg. MoE @32 (trained)",
        ),
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        frameon=True, fontsize=20,
        handletextpad=0.7, borderpad=0.8,
        facecolor="white", edgecolor="#CCCCCC",
        labelcolor=INK_COLOR,
    )
    leg.get_frame().set_linewidth(1.0)


def _register_pipeline_fonts() -> None:
    """Register the EB Garamond / IBM Plex Mono TTFs (extracted from the
    eval_pipeline bundle) with matplotlib, if they exist on disk."""
    if not _FONT_DIR.is_dir():
        return
    for ttf in _FONT_DIR.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass


def _add_clip_to_page(page, x: float, y: float, w: float, h: float) -> None:
    """Wrap the page's content stream in a graphics-state save + clip + restore.

    Subsequent draws (and any merge that appends to this page's contents)
    only render within the rectangle (x, y, x+w, y+h) in the source's
    page coordinate system.
    """
    contents = page.get_contents()
    if contents is None:
        return
    raw = contents.get_data()
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    prefix = f"q\n{x:.3f} {y:.3f} {w:.3f} {h:.3f} re W n\n".encode("latin-1")
    suffix = b"\nQ\n"
    new_stream = DecodedStreamObject()
    new_stream.set_data(prefix + raw + suffix)
    page[NameObject("/Contents")] = new_stream


def _render_pareto_to_pdf(df: pd.DataFrame, w_in: float, h_in: float, out_path: Path) -> None:
    """Render the Pareto plot to a vector PDF at exactly (w_in, h_in)."""
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    _draw_pareto(ax, df)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def render_combined(pipeline_pdf: Path, pareto_csv: Path, output: Path,
                    pareto_w_in: float, pareto_h_in: float, gap_in: float,
                    eval_crop_left_in: float, eval_crop_right_in: float) -> None:
    df = pd.read_csv(pareto_csv)

    pdf_eval = PdfReader(str(pipeline_pdf))
    eval_page = pdf_eval.pages[0]
    eval_mb = eval_page.mediabox
    eval_w_pt = float(eval_mb.width)
    eval_h_pt = float(eval_mb.height)

    crop_left_pt = eval_crop_left_in * PT_PER_IN
    crop_right_pt = eval_crop_right_in * PT_PER_IN
    effective_eval_w_pt = eval_w_pt - crop_left_pt - crop_right_pt

    # Clip the eval page's content stream to the visible-after-crop rectangle.
    _add_clip_to_page(
        eval_page,
        x=float(eval_mb.left) + crop_left_pt,
        y=float(eval_mb.bottom),
        w=effective_eval_w_pt,
        h=eval_h_pt,
    )

    pareto_tmp = output.with_suffix(".pareto.tmp.pdf")
    _render_pareto_to_pdf(df, pareto_w_in, pareto_h_in, pareto_tmp)

    pdf_par = PdfReader(str(pareto_tmp))
    par_page = pdf_par.pages[0]
    par_mb = par_page.mediabox
    par_w_pt = float(par_mb.width)
    par_h_pt = float(par_mb.height)

    gap_pt = gap_in * PT_PER_IN
    total_w_pt = effective_eval_w_pt + gap_pt + par_w_pt
    total_h_pt = max(eval_h_pt, par_h_pt)

    writer = PdfWriter()
    blank = writer.add_blank_page(width=total_w_pt, height=total_h_pt)

    eval_y_off = (total_h_pt - eval_h_pt) / 2.0
    op_eval = Transformation().translate(
        tx=-float(eval_mb.left) - crop_left_pt,
        ty=-float(eval_mb.bottom) + eval_y_off,
    )
    blank.merge_transformed_page(eval_page, op_eval)

    par_y_off = (total_h_pt - par_h_pt) / 2.0
    par_x_off = effective_eval_w_pt + gap_pt
    op_par = Transformation().translate(
        tx=-float(par_mb.left) + par_x_off,
        ty=-float(par_mb.bottom) + par_y_off,
    )
    blank.merge_transformed_page(par_page, op_par)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        writer.write(f)

    pareto_tmp.unlink()
    print(
        f"Wrote {output}  "
        f"({total_w_pt / PT_PER_IN:.2f}in x {total_h_pt / PT_PER_IN:.2f}in)"
    )


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams["text.color"] = INK_COLOR
    mpl.rcParams["axes.labelcolor"] = INK_COLOR
    mpl.rcParams["xtick.color"] = INK_COLOR
    mpl.rcParams["ytick.color"] = INK_COLOR
    render_combined(
        args.pipeline_pdf, args.pareto_csv, args.output,
        pareto_w_in=args.pareto_width_in,
        pareto_h_in=args.pareto_height_in,
        gap_in=args.gap_in,
        eval_crop_left_in=args.eval_crop_left_in,
        eval_crop_right_in=args.eval_crop_right_in,
    )


if __name__ == "__main__":
    main()
