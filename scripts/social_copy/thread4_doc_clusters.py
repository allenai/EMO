#!/usr/bin/env python3
"""Thread 4 figure: top-cluster cards for Standard MoE vs EMO.

Adapted from ``scripts/other_figures/plot_modmoe_vs_stdmoe_doc_clusters.py``
and ``figure_token_cluster_comparison.py``: renders ONLY the two cluster-list
cards (no per-document zoom panel). Because there's no chosen document on
display, no row is highlighted — every cluster is shown in the same neutral
style.

Output: claude_outputs/social_post/thread4_doc_clusters.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_HTML = (
    REPO_ROOT / "claude_outputs" / "clustering" / "pretraining"
    / "modmoe_vs_stdmoe_comparison.html"
)
OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "social_post"
OUTPUT = OUTPUT_DIR / "thread4_doc_clusters.png"

# Match the paper figure: Top 18 / 32 clusters per model.
TOP_N_CLUSTERS = 18

HEADER_TITLE = "Standard MoEs cluster by syntax. EMO clusters by domain."
HEADER_SUBTITLE = "Top routing-pattern clusters from 12K pretraining docs."

# Editorial palette (copied from the upstream paper figure).
FG = "#1a202c"
FG_MUTED = "#556070"
FG_FAINT = "#9aa4b2"
DOT_DIM = "#cbd5e0"
BG_CARD = "#ffffff"
BG_PAGE = "#fafbfc"
BORDER = "#e5e8ec"
DIVIDER = "#eef0f3"

STDMOE_ACCENT = {
    "accent":      "#5B8E3F",
    "accent_dark": "#225C2E",
}
MODMOE_ACCENT = {
    "accent":      "#B8327C",
    "accent_dark": "#3F1052",
}


def extract_js_const(html: str, name: str) -> Any:
    start = html.find(f"const {name} = ")
    if start == -1:
        raise KeyError(name)
    start += len(f"const {name} = ")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(html)):
        ch = html[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start : i + 1])
    raise ValueError(f"Unterminated const {name}")


def add_rounded_card(
    ax, x: float, y: float, w: float, h: float,
    facecolor: str, edgecolor: str, lw: float = 1.0, rounding: float = 0.025,
) -> None:
    box = FancyBboxPatch(
        (x + rounding, y + rounding),
        w - 2 * rounding, h - 2 * rounding,
        boxstyle=f"round,pad={rounding},rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        transform=ax.transAxes, zorder=0,
    )
    ax.add_patch(box)


def render_cluster_card(
    ax, clusters: List[dict], total_tokens: int,
    label: str, accent: str, top_n: int,
) -> None:
    """Render a single model's cluster list. No row highlight."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_rounded_card(ax, 0.0, 0.0, 1.0, 1.0, BG_CARD, BORDER, lw=0.8)

    # Top accent stripe.
    ax.add_patch(patches.Rectangle(
        (0, 0.96), 1, 0.04,
        facecolor=accent, edgecolor="none",
        transform=ax.transAxes, zorder=1,
    ))

    # Model label.
    ax.text(
        0.05, 0.91, label,
        fontsize=22, fontweight="bold", color=FG,
        transform=ax.transAxes, ha="left", va="top",
        family="sans-serif",
    )

    # Header divider + section label.
    ax.plot(
        [0.05, 0.95], [0.825, 0.825],
        color=DIVIDER, linewidth=1.0,
        transform=ax.transAxes, zorder=1,
    )
    ax.text(
        0.05, 0.806, "TOP CLUSTERS  ·  BY TOKEN COUNT",
        fontsize=10, color=FG_FAINT, fontweight="bold",
        transform=ax.transAxes, ha="left", va="top",
        family="sans-serif",
    )

    sorted_clusters = sorted(clusters, key=lambda c: -c["size"])[:top_n]
    y_top = 0.775
    y_bot = 0.055
    row_h = (y_top - y_bot) / top_n

    for i, c in enumerate(sorted_clusters):
        y = y_top - (i + 0.5) * row_h
        pct = c["size"] / total_tokens * 100

        # Neutral row dot.
        ax.add_patch(patches.Circle(
            (0.075, y), row_h * 0.17,
            facecolor=DOT_DIM, edgecolor="none",
            transform=ax.transAxes, zorder=3,
        ))

        label_short = c["label"] if len(c["label"]) <= 38 else c["label"][:36] + "…"
        ax.text(
            0.115, y, label_short,
            fontsize=12, fontweight="normal", color=FG,
            transform=ax.transAxes, va="center",
            family="sans-serif",
        )
        ax.text(
            0.955, y, f"{pct:4.1f}%",
            fontsize=11, color=FG_MUTED, fontweight="normal",
            transform=ax.transAxes, va="center", ha="right",
            family="monospace",
        )

    # Footer.
    ax.plot(
        [0.05, 0.95], [0.045, 0.045],
        color=DIVIDER, linewidth=1.0,
        transform=ax.transAxes, zorder=1,
    )
    ax.text(
        0.5, 0.022,
        f"Top {top_n} of {len(clusters)} clusters",
        fontsize=9.5, color=FG_FAINT, style="italic",
        transform=ax.transAxes, ha="center", va="center",
    )


def main() -> None:
    if not SOURCE_HTML.exists():
        print(f"ERROR: source HTML not found at {SOURCE_HTML}", file=sys.stderr)
        sys.exit(1)

    with open(SOURCE_HTML) as f:
        html = f.read()
    M1 = extract_js_const(html, "M1")  # EMO
    M2 = extract_js_const(html, "M2")  # Standard MoE

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": BORDER,
        "savefig.facecolor": BG_PAGE,
        "figure.facecolor": BG_PAGE,
    })

    fig = plt.figure(figsize=(15, 8.5))
    fig.patch.set_facecolor(BG_PAGE)

    # Header room (suptitle + subtitle), then two side-by-side cluster cards.
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1, 1],
        wspace=0.05,
        left=0.03, right=0.97,
        top=0.86, bottom=0.04,
    )

    # Left: Standard MoE; Right: EMO. Same order as the upstream paper figure.
    ax_left = fig.add_subplot(gs[0, 0])
    render_cluster_card(
        ax_left,
        clusters=M2["clusters"],
        total_tokens=sum(c["size"] for c in M2["clusters"]),
        label="Standard MoE",
        accent=STDMOE_ACCENT["accent"],
        top_n=TOP_N_CLUSTERS,
    )

    ax_right = fig.add_subplot(gs[0, 1])
    render_cluster_card(
        ax_right,
        clusters=M1["clusters"],
        total_tokens=sum(c["size"] for c in M1["clusters"]),
        label="EMO",
        accent=MODMOE_ACCENT["accent"],
        top_n=TOP_N_CLUSTERS,
    )

    # Suptitle + subtitle (baked-in caption for the social post).
    fig.text(
        0.5, 0.965,
        HEADER_TITLE,
        fontsize=20, fontweight="bold", color=FG,
        ha="center", va="top", family="sans-serif",
    )
    fig.text(
        0.5, 0.918,
        HEADER_SUBTITLE,
        fontsize=12, color=FG_MUTED, style="italic",
        ha="center", va="top", family="sans-serif",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=BG_PAGE)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
