#!/usr/bin/env python3
"""Research-paper figure comparing token-level clustering between MOSE
(twolevel) and the baseline MoE, in an editorial/infographic style.

Layout:
  - Top: two card-style panels listing the top-N clusters for each model
  - Bottom: "zoom-in" showing the same document with one cluster highlighted
           per model (MOSE's top semantic cluster vs baseline's top syntactic
           cluster)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================================================================
# CONFIG
# ============================================================================

BASE = Path(
    "/Users/ryanyxw/Desktop/Berkeley/repos/phdbrainstorm/FlexMoE/"
    "claude_outputs/analysis/router_clustering_pretraining_shuffled_token_truncated"
)
CLUSTER_SUBDIR = "token_probs_mean_pca_l2_spherical_kmeans_k64"

# ---- Model specs (first entry goes LEFT) ----
MODELS = [
    (
        "twolevel",
        {
            "path": "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301",
            "label": "MOSE",
            "sublabel": "(twolevel MoE)",
            "subtitle": "learns topical / semantic clusters",
            "highlight_cluster_id": 18,  # "Sports & athletics"
            "accent": "#0e7c7b",  # deep teal
            "accent_soft": "#d3e9e8",  # soft teal bg
            "accent_dark": "#054e4c",
        },
    ),
    (
        "baseline",
        {
            "path": "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211",
            "label": "Standard MoE",
            "sublabel": "(baseline)",
            "subtitle": "learns syntactic / function-word clusters",
            "highlight_cluster_id": 34,  # "Conditional & concessive clauses"
            "accent": "#b45309",  # burnt amber
            "accent_soft": "#f5e6d0",
            "accent_dark": "#5c2a00",
        },
    ),
]

TARGET_DOC_INDEX = 9090  # Lakers / NBA sports article
TOP_N_CLUSTERS = 22

# ---- Typography & palette ----
FG = "#1a202c"  # near-black
FG_MUTED = "#556070"  # medium slate
FG_FAINT = "#9aa4b2"  # light slate
DOT_DIM = "#cbd5e0"  # inactive dot
BG_CARD = "#ffffff"
BG_PAGE = "#fafbfc"
BORDER = "#e5e8ec"
DIVIDER = "#eef0f3"

OUTPUT_PATH = Path(__file__).resolve().parent / "figure_token_cluster_comparison.png"


# ============================================================================
# DATA
# ============================================================================


def extract_js_const(html: str, name: str):
    start = html.find(f"const {name} = ")
    if start == -1:
        raise KeyError(name)
    start += len(f"const {name} = ")
    depth = 0
    in_str = False
    esc = False
    assert html[start] in "[{"
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
    raise ValueError("Unterminated const")


def load_model_data(model_key: str, spec: dict):
    html_path = BASE / spec["path"] / CLUSTER_SUBDIR / "cluster_explorer.html"
    print(f"[{model_key}] Loading {html_path.name}")
    with open(html_path) as f:
        html = f.read()
    clusters = extract_js_const(html, "CLUSTERS")
    doc_texts = extract_js_const(html, "DOC_TEXTS")
    target = next((d for d in doc_texts if d["di"] == TARGET_DOC_INDEX), None)
    if target is None:
        raise RuntimeError(f"Doc #{TARGET_DOC_INDEX} not found in {model_key}")
    return {
        "clusters": clusters,
        "cluster_by_id": {c["id"]: c for c in clusters},
        "doc": target,
        "total_tokens": sum(c["size"] for c in clusters),
    }


# ============================================================================
# HELPERS
# ============================================================================


def hex_to_rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


def wrap_tokens_into_lines(tokens, chars_per_line, max_lines):
    row, col = 0, 0
    placements = []
    for tok in tokens:
        text = tok["t"]
        cid = tok["c"]
        parts = text.split("\n")
        for pi, part in enumerate(parts):
            if pi > 0:
                row += 1
                col = 0
                if row >= max_lines:
                    return placements
            if not part:
                continue
            remaining = part
            while remaining:
                space = chars_per_line - col
                if len(remaining) <= space:
                    placements.append((row, col, remaining, cid))
                    col += len(remaining)
                    remaining = ""
                else:
                    if space > 0:
                        placements.append((row, col, remaining[:space], cid))
                        remaining = remaining[space:]
                    row += 1
                    col = 0
                    if row >= max_lines:
                        return placements
    return placements


def add_rounded_card(ax, x, y, w, h, facecolor, edgecolor, lw=1.0, rounding=0.025):
    """Add a FancyBboxPatch rounded card in figure-coord space."""
    box = FancyBboxPatch(
        (x + rounding, y + rounding),
        w - 2 * rounding,
        h - 2 * rounding,
        boxstyle=f"round,pad={rounding},rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(box)


# ============================================================================
# RENDERING
# ============================================================================


def render_cluster_card(ax, clusters, total_tokens, spec, top_n):
    """Render one model's cluster list as a polished card."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    accent = spec["accent"]
    accent_soft = spec["accent_soft"]
    accent_dark = spec["accent_dark"]
    hid = spec["highlight_cluster_id"]

    # Card background
    add_rounded_card(ax, 0.0, 0.0, 1.0, 1.0, BG_CARD, BORDER, lw=0.8)

    # Accent stripe at the top
    stripe = patches.Rectangle(
        (0, 0.96),
        1,
        0.04,
        facecolor=accent,
        edgecolor="none",
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(stripe)

    # Model label (big) + sublabel (small, to the right)
    ax.text(
        0.05,
        0.91,
        spec["label"],
        fontsize=18,
        fontweight="bold",
        color=FG,
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="sans-serif",
    )
    ax.text(
        0.05,
        0.862,
        spec["sublabel"],
        fontsize=10,
        color=FG_MUTED,
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="sans-serif",
    )

    # Tagline
    ax.text(
        0.95,
        0.89,
        spec["subtitle"],
        fontsize=10.5,
        color=accent_dark,
        style="italic",
        transform=ax.transAxes,
        ha="right",
        va="top",
        family="sans-serif",
    )

    # Divider under header
    ax.plot(
        [0.05, 0.95], [0.825, 0.825], color=DIVIDER, linewidth=1.0, transform=ax.transAxes, zorder=1
    )

    # Section label
    ax.text(
        0.05,
        0.806,
        "TOP CLUSTERS  ·  BY TOKEN COUNT",
        fontsize=8,
        color=FG_FAINT,
        fontweight="bold",
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="sans-serif",
    )

    # Cluster rows
    sorted_clusters = sorted(clusters, key=lambda c: -c["size"])[:top_n]

    y_top = 0.775
    y_bot = 0.055
    row_h = (y_top - y_bot) / top_n

    for i, c in enumerate(sorted_clusters):
        y = y_top - (i + 0.5) * row_h
        label = c["label"]
        pct = c["size"] / total_tokens * 100
        is_highlight = c["id"] == hid

        # Highlighted row: rounded soft background + left accent bar
        if is_highlight:
            bg = FancyBboxPatch(
                (0.035, y - row_h * 0.42),
                0.93,
                row_h * 0.84,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor=accent_soft,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(bg)
            # Left accent bar
            bar = FancyBboxPatch(
                (0.038, y - row_h * 0.38),
                0.009,
                row_h * 0.76,
                boxstyle="round,pad=0,rounding_size=0.003",
                facecolor=accent,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=2,
            )
            ax.add_patch(bar)

        # Dot marker
        dot_color = accent if is_highlight else DOT_DIM
        ax.add_patch(
            patches.Circle(
                (0.07, y),
                row_h * 0.17,
                facecolor=dot_color,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=3,
            )
        )

        # Cluster ID
        ax.text(
            0.105,
            y,
            f"C{c['id']:02d}",
            fontsize=7,
            color=FG_FAINT,
            transform=ax.transAxes,
            va="center",
            family="monospace",
        )

        # Label
        label_short = label if len(label) <= 42 else label[:40] + "…"
        text_color = accent_dark if is_highlight else FG
        weight = "bold" if is_highlight else "normal"
        ax.text(
            0.17,
            y,
            label_short,
            fontsize=9.5,
            fontweight=weight,
            color=text_color,
            transform=ax.transAxes,
            va="center",
            family="sans-serif",
        )

        # Percentage
        pct_color = accent_dark if is_highlight else FG_MUTED
        ax.text(
            0.955,
            y,
            f"{pct:4.1f}%",
            fontsize=8.5,
            color=pct_color,
            fontweight="bold" if is_highlight else "normal",
            transform=ax.transAxes,
            va="center",
            ha="right",
            family="monospace",
        )

    # Footer
    total_shown = sum(c["size"] for c in sorted_clusters)
    shown_pct = total_shown / total_tokens * 100
    ax.plot(
        [0.05, 0.95], [0.045, 0.045], color=DIVIDER, linewidth=1.0, transform=ax.transAxes, zorder=1
    )
    ax.text(
        0.5,
        0.022,
        f"Top {top_n} of {len(clusters)} clusters  ·  {shown_pct:.0f}% of all tokens",
        fontsize=7.5,
        color=FG_FAINT,
        style="italic",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def render_doc_zoom_card(ax, tokens, spec, cluster_label, target_pct):
    """Render the zoomed-in document as a polished card."""
    ax.axis("off")

    chars_per_line = 72
    max_lines = 10
    header_h = 1.4
    total_h = max_lines + header_h
    ax.set_xlim(-1.5, chars_per_line + 1.5)
    ax.set_ylim(-0.5, total_h)

    accent = spec["accent"]
    accent_soft = spec["accent_soft"]
    accent_dark = spec["accent_dark"]
    hid = spec["highlight_cluster_id"]

    # Card background (rounded corners)
    card = FancyBboxPatch(
        (-0.8, 0.1),
        chars_per_line + 1.6,
        total_h - 0.1,
        boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=BG_CARD,
        edgecolor=BORDER,
        linewidth=0.9,
        zorder=0,
    )
    ax.add_patch(card)

    # Accent pill on the left side of the header
    pill = FancyBboxPatch(
        (0.2, total_h - 1.05),
        0.7,
        0.6,
        boxstyle="round,pad=0,rounding_size=0.3",
        facecolor=accent,
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(pill)

    # Header: cluster label + model name + percentage
    header_y = total_h - 0.75
    ax.text(
        1.5,
        header_y,
        f"C{hid:02d}  ·  {cluster_label}",
        fontsize=11,
        fontweight="bold",
        color=accent_dark,
        ha="left",
        va="center",
        family="sans-serif",
        zorder=3,
    )
    ax.text(
        chars_per_line + 0.4,
        header_y,
        f"{spec['label']}   ·   {target_pct:.0f}% of doc",
        fontsize=10,
        color=FG_MUTED,
        ha="right",
        va="center",
        family="sans-serif",
        zorder=3,
    )

    # Divider line between header and document
    ax.plot(
        [0.5, chars_per_line - 0.5],
        [max_lines + 0.1, max_lines + 0.1],
        color=DIVIDER,
        linewidth=0.8,
        zorder=2,
    )

    # --- Render document tokens ---
    placements = wrap_tokens_into_lines(tokens, chars_per_line, max_lines)

    # Background highlights for target-cluster tokens (rounded pills)
    for row, col, text, cid in placements:
        y = max_lines - row - 0.55
        if cid == hid:
            pill = FancyBboxPatch(
                (col - 0.08, y - 0.3),
                len(text) + 0.16,
                0.62,
                boxstyle="round,pad=0,rounding_size=0.22",
                facecolor=accent_soft,
                edgecolor=accent,
                linewidth=0.45,
                zorder=2,
            )
            ax.add_patch(pill)

    # Text pass
    for row, col, text, cid in placements:
        y = max_lines - row - 0.55
        if cid == hid:
            color = accent_dark
            weight = "bold"
        else:
            color = "#b4bac4"
            weight = "normal"
        ax.text(
            col,
            y,
            text,
            fontsize=8,
            family="monospace",
            color=color,
            fontweight=weight,
            va="center",
            ha="left",
            zorder=3,
        )


# ============================================================================
# MAIN
# ============================================================================


def main():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": BORDER,
            "savefig.facecolor": BG_PAGE,
            "figure.facecolor": BG_PAGE,
        }
    )

    data = {key: load_model_data(key, spec) for key, spec in MODELS}

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG_PAGE)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1, 1],
        height_ratios=[2.4, 1.0],
        hspace=0.14,
        wspace=0.05,
        left=0.03,
        right=0.97,
        top=0.905,
        bottom=0.035,
    )

    # ---- Top row: cluster cards ----
    for col_idx, (key, spec) in enumerate(MODELS):
        d = data[key]
        ax = fig.add_subplot(gs[0, col_idx])
        render_cluster_card(
            ax,
            clusters=d["clusters"],
            total_tokens=d["total_tokens"],
            spec=spec,
            top_n=TOP_N_CLUSTERS,
        )

    # ---- Bottom row: document zoom-in cards ----
    for col_idx, (key, spec) in enumerate(MODELS):
        d = data[key]
        tokens = d["doc"]["tokens"]
        hid = spec["highlight_cluster_id"]
        in_cluster = sum(1 for t in tokens if t["c"] == hid)
        pct = in_cluster / len(tokens) * 100
        cluster_label = d["cluster_by_id"][hid]["label"]

        ax = fig.add_subplot(gs[1, col_idx])
        render_doc_zoom_card(
            ax, tokens=tokens, spec=spec, cluster_label=cluster_label, target_pct=pct
        )

    # ---- Suptitle block ----
    fig.text(
        0.5,
        0.965,
        "Token-level router clusters in MOSE vs. a standard MoE",
        fontsize=19,
        fontweight="bold",
        color=FG,
        ha="center",
        va="top",
        family="sans-serif",
    )
    fig.text(
        0.5,
        0.932,
        "Clustering router softmax probabilities at k=64 reveals how each model organizes experts.",
        fontsize=11,
        color=FG_MUTED,
        style="italic",
        ha="center",
        va="top",
        family="sans-serif",
    )

    # Section label between the two rows
    fig.text(
        0.5,
        0.3,
        f"Each model's top cluster for the same NBA article  ·  Doc #{TARGET_DOC_INDEX}",
        fontsize=10,
        color=FG_MUTED,
        style="italic",
        ha="center",
        va="center",
        family="sans-serif",
    )

    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG_PAGE)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
