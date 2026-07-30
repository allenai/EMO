#!/usr/bin/env python3
"""Editorial figure comparing EMO vs Standard MoE token-level clustering
on a single representative document.

Sibling of ``figure_token_cluster_comparison.py`` but reads from the
combined comparison page ``claude_outputs/clustering/pretraining/modmoe_vs_stdmoe_comparison.html``.

Picks a document where:
    - EMO concentrates ~96% of the document's tokens into ONE *domain*
      cluster ("Health, medical & wellness"). The few divergent tokens
      cluster at the very start of the document (first sentence).
    - Standard MoE's most-common cluster on the same document is a pure
      function-word *syntax* pattern ("Possessives & definite articles:
      the/my/your"), and the document is possessive-heavy so the syntactic
      pattern is highly visible in the rendered text.
    - The document has zero internal newlines, so the document-zoom panel
      fills with continuous prose with no whitespace breaks.
    - Both highlight clusters live in their model's top-18 by global token
      size, so they appear in the cluster cards naturally without pinning.

The default target is doc#1624 (DCLM, abdominal-exercise advice). Pass
--target-doc-id to swap in a different document; pass -1 to auto-select
via an offline scan.

Output: claude_outputs/other_figures/modmoe_vs_stdmoe_doc_clusters.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, List

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HTML = (
    REPO_ROOT / "claude_outputs" / "clustering" / "pretraining" / "modmoe_vs_stdmoe_comparison.html"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "claude_outputs" / "other_figures" / "modmoe_vs_stdmoe_doc_clusters.pdf"
)

# Pre-selected via an offline scan: high EMO concentration in a topical
# cluster, with Standard MoE's top cluster being a syntactic surface pattern.
DEFAULT_TARGET_DOC_ID = 1624
TOP_N_CLUSTERS = 18  # fewer rows so each row has room for the bigger label

# Editorial palette (matches figure_token_cluster_comparison.py).
FG = "#1a202c"
FG_MUTED = "#556070"
FG_FAINT = "#9aa4b2"
DOT_DIM = "#cbd5e0"
BG_CARD = "#ffffff"
BG_PAGE = "#fafbfc"
BORDER = "#e5e8ec"
DIVIDER = "#eef0f3"

# Per-model accent palette — matches the paper figures in
# scripts/ryanwang/pruning_plots/paper_figure_codes (Reg. MoE = green
# family, EMO = magenta/pink family). The accent_soft tints are the
# same green / pink legend washes used in main_results_abs_bars_1t.pdf.
STDMOE_ACCENT = {
    "accent": "#5B8E3F",  # REG_PALETTE[1]  — medium green
    "accent_soft": "#EEF6E4",  # green wash used in legend backgrounds
    "accent_dark": "#225C2E",  # REG_PALETTE[0]  — dark forest green
}
MODMOE_ACCENT = {
    "accent": "#B8327C",  # FLEX_PALETTE[2] — EMO magenta
    "accent_soft": "#FBE7EF",  # pink wash used in legend backgrounds
    "accent_dark": "#3F1052",  # FLEX_PALETTE[0] — deep purple
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def extract_js_const(html: str, name: str) -> Any:
    """Brace-balanced extraction of `const NAME = {...}` or `[...]` JSON."""
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


def auto_select_doc(
    docs: List[dict],
    m1_clusters: List[dict],
    m2_clusters: List[dict],
    top_n: int,
) -> int:
    """Pick the doc with the strongest EMO-concentrates-semantic /
    StdMoE-concentrates-syntactic contrast, additionally requiring both
    of the doc's top-cluster IDs to live in their model's global top-N
    by size — so they show up in the rendered cluster cards naturally
    without any pinning."""
    # Pure function-word / punctuation StdMoE clusters only.
    SYNTACTIC_M2 = {2, 5, 6, 8, 10, 13, 15, 19, 20, 21, 22, 27}
    # Only count EMO clusters in clear topical-domain categories.
    DOMAIN_M1_CATS = {
        "arts",
        "business",
        "code",
        "education",
        "health",
        "news",
        "science",
    }

    m1_top_ids = {c["id"] for c in sorted(m1_clusters, key=lambda c: -c["size"])[:top_n]}
    m2_top_ids = {c["id"] for c in sorted(m2_clusters, key=lambda c: -c["size"])[:top_n]}
    m1_by_id = {c["id"]: c for c in m1_clusters}

    best = None
    best_score = -1.0
    for d in docs:
        n = len(d["t"])
        if n < 80 or n > 600:
            continue
        c1_top, c1_n = Counter(d["c1"]).most_common(1)[0]
        c2_top, c2_n = Counter(d["c2"]).most_common(1)[0]
        if c1_top not in m1_top_ids or c2_top not in m2_top_ids:
            continue
        if m1_by_id[c1_top].get("category") not in DOMAIN_M1_CATS:
            continue
        if c2_top not in SYNTACTIC_M2:
            continue
        m1_pct = c1_n / n
        m2_pct = c2_n / n
        if m1_pct < 0.55 or m2_pct < 0.15:
            continue
        score = m1_pct + m2_pct
        if score > best_score:
            best_score = score
            best = d
    if best is None:
        raise RuntimeError("No suitable document found")
    return best["di"]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def add_rounded_card(ax, x, y, w, h, facecolor, edgecolor, lw=1.0, rounding=0.025):
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


def wrap_tokens_into_lines(tokens: List[str], chars_per_line: int, max_lines: int):
    """Place each token on a (row, col) grid. tokens is a list of strings."""
    row, col = 0, 0
    placements = []
    for ti, text in enumerate(tokens):
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
                    placements.append((row, col, remaining, ti))
                    col += len(remaining)
                    remaining = ""
                else:
                    if space > 0:
                        placements.append((row, col, remaining[:space], ti))
                        remaining = remaining[space:]
                    row += 1
                    col = 0
                    if row >= max_lines:
                        return placements
    return placements


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_cluster_card(ax, model_data: dict, spec: dict, top_n: int):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    accent = spec["accent"]
    accent_soft = spec["accent_soft"]
    accent_dark = spec["accent_dark"]
    hid = spec["highlight_cluster_id"]

    add_rounded_card(ax, 0.0, 0.0, 1.0, 1.0, BG_CARD, BORDER, lw=0.8)

    # Top accent stripe.
    ax.add_patch(
        patches.Rectangle(
            (0, 0.96),
            1,
            0.04,
            facecolor=accent,
            edgecolor="none",
            transform=ax.transAxes,
            zorder=1,
        )
    )

    ax.text(
        0.05,
        0.91,
        spec["label"],
        fontsize=22,
        fontweight="bold",
        color=FG,
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="sans-serif",
    )

    ax.plot(
        [0.05, 0.95], [0.825, 0.825], color=DIVIDER, linewidth=1.0, transform=ax.transAxes, zorder=1
    )
    ax.text(
        0.05,
        0.806,
        "TOP CLUSTERS  ·  BY TOKEN COUNT",
        fontsize=10,
        color=FG_FAINT,
        fontweight="bold",
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="sans-serif",
    )

    clusters = model_data["clusters"]
    total_tokens = model_data["total_tokens"]
    sorted_clusters = sorted(clusters, key=lambda c: -c["size"])[:top_n]

    y_top = 0.775
    y_bot = 0.055
    row_h = (y_top - y_bot) / top_n

    for i, c in enumerate(sorted_clusters):
        y = y_top - (i + 0.5) * row_h
        is_highlight = c["id"] == hid
        pct = c["size"] / total_tokens * 100

        if is_highlight:
            ax.add_patch(
                FancyBboxPatch(
                    (0.035, y - row_h * 0.42),
                    0.93,
                    row_h * 0.84,
                    boxstyle="round,pad=0,rounding_size=0.012",
                    facecolor=accent_soft,
                    edgecolor="none",
                    transform=ax.transAxes,
                    zorder=1,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (0.038, y - row_h * 0.38),
                    0.009,
                    row_h * 0.76,
                    boxstyle="round,pad=0,rounding_size=0.003",
                    facecolor=accent,
                    edgecolor="none",
                    transform=ax.transAxes,
                    zorder=2,
                )
            )

        ax.add_patch(
            patches.Circle(
                (0.075, y),
                row_h * 0.17,
                facecolor=accent if is_highlight else DOT_DIM,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=3,
            )
        )

        label_short = c["label"] if len(c["label"]) <= 38 else c["label"][:36] + "…"
        ax.text(
            0.115,
            y,
            label_short,
            fontsize=12,
            fontweight="bold" if is_highlight else "normal",
            color=accent_dark if is_highlight else FG,
            transform=ax.transAxes,
            va="center",
            family="sans-serif",
        )
        ax.text(
            0.955,
            y,
            f"{pct:4.1f}%",
            fontsize=11,
            color=accent_dark if is_highlight else FG_MUTED,
            fontweight="bold" if is_highlight else "normal",
            transform=ax.transAxes,
            va="center",
            ha="right",
            family="monospace",
        )

    ax.plot(
        [0.05, 0.95], [0.045, 0.045], color=DIVIDER, linewidth=1.0, transform=ax.transAxes, zorder=1
    )
    ax.text(
        0.5,
        0.022,
        f"Top {top_n} of {len(clusters)} clusters",
        fontsize=9.5,
        color=FG_FAINT,
        style="italic",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def render_doc_zoom_card(
    ax,
    doc_tokens: List[str],
    doc_cluster_ids: List[int],
    spec: dict,
    cluster_label: str,
    target_pct: float,
):
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

    header_y = total_h - 0.75
    ax.text(
        1.5,
        header_y,
        cluster_label,
        fontsize=14,
        fontweight="bold",
        color=accent_dark,
        ha="left",
        va="center",
        family="sans-serif",
        zorder=3,
    )

    ax.plot(
        [0.5, chars_per_line - 0.5],
        [max_lines + 0.1, max_lines + 0.1],
        color=DIVIDER,
        linewidth=0.8,
        zorder=2,
    )

    placements = wrap_tokens_into_lines(doc_tokens, chars_per_line, max_lines)

    for row, col, text, ti in placements:
        y = max_lines - row - 0.55
        if doc_cluster_ids[ti] == hid:
            ax.add_patch(
                FancyBboxPatch(
                    (col - 0.08, y - 0.3),
                    len(text) + 0.16,
                    0.62,
                    boxstyle="round,pad=0,rounding_size=0.22",
                    facecolor=accent_soft,
                    edgecolor=accent,
                    linewidth=0.45,
                    zorder=2,
                )
            )

    for row, col, text, ti in placements:
        y = max_lines - row - 0.55
        if doc_cluster_ids[ti] == hid:
            color, weight = accent_dark, "bold"
        else:
            color, weight = "#b4bac4", "normal"
        ax.text(
            col,
            y,
            text,
            fontsize=10,
            family="monospace",
            color=color,
            fontweight=weight,
            va="center",
            ha="left",
            zorder=3,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-doc-id",
        type=int,
        default=DEFAULT_TARGET_DOC_ID,
        help="Document index `di` to render. Pass -1 to auto-select.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.html}")
    with open(args.html) as f:
        html = f.read()
    M1 = extract_js_const(html, "M1")
    M2 = extract_js_const(html, "M2")
    DOCS = extract_js_const(html, "DOCS")

    m1_by_id = {c["id"]: c for c in M1["clusters"]}
    m2_by_id = {c["id"]: c for c in M2["clusters"]}

    target_id = args.target_doc_id
    if target_id == -1:
        target_id = auto_select_doc(DOCS, M1["clusters"], M2["clusters"], TOP_N_CLUSTERS)
        print(f"Auto-selected doc#{target_id}")

    target_doc = next((d for d in DOCS if d["di"] == target_id), None)
    if target_doc is None:
        raise RuntimeError(f"Doc #{target_id} not found")
    n_tokens = len(target_doc["t"])

    # Top clusters per model on this doc.
    m1_top, m1_top_n = Counter(target_doc["c1"]).most_common(1)[0]
    m2_top, m2_top_n = Counter(target_doc["c2"]).most_common(1)[0]
    m1_top_pct = m1_top_n / n_tokens * 100
    m2_top_pct = m2_top_n / n_tokens * 100

    print(f"Doc #{target_id}  ({target_doc['s']}, {n_tokens} tokens)")
    print(
        f"  EMO       top: C{m1_top:02d} {m1_by_id[m1_top]['label']!r}  "
        f"{m1_top_pct:.1f}% of tokens"
    )
    print(
        f"  Standard MoE top: C{m2_top:02d} {m2_by_id[m2_top]['label']!r}  "
        f"{m2_top_pct:.1f}% of tokens"
    )

    models = [
        (
            "stdmoe",
            {
                "model_data": {
                    "clusters": M2["clusters"],
                    "total_tokens": sum(c["size"] for c in M2["clusters"]),
                },
                "doc_cluster_ids": target_doc["c2"],
                "label": "Standard MoE",
                "highlight_cluster_id": m2_top,
                "highlight_cluster_label": m2_by_id[m2_top]["label"],
                "highlight_cluster_pct": m2_top_pct,
                **STDMOE_ACCENT,
            },
        ),
        (
            "modmoe",
            {
                "model_data": {
                    "clusters": M1["clusters"],
                    "total_tokens": sum(c["size"] for c in M1["clusters"]),
                },
                "doc_cluster_ids": target_doc["c1"],
                "label": "EMO",
                "highlight_cluster_id": m1_top,
                "highlight_cluster_label": m1_by_id[m1_top]["label"],
                "highlight_cluster_pct": m1_top_pct,
                **MODMOE_ACCENT,
            },
        ),
    ]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": BORDER,
            "savefig.facecolor": BG_PAGE,
            "figure.facecolor": BG_PAGE,
        }
    )

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
        top=0.965,
        bottom=0.035,
    )

    for col_idx, (_key, spec) in enumerate(models):
        ax = fig.add_subplot(gs[0, col_idx])
        render_cluster_card(ax, spec["model_data"], spec, TOP_N_CLUSTERS)

    for col_idx, (_key, spec) in enumerate(models):
        ax = fig.add_subplot(gs[1, col_idx])
        render_doc_zoom_card(
            ax,
            doc_tokens=target_doc["t"],
            doc_cluster_ids=spec["doc_cluster_ids"],
            spec=spec,
            cluster_label=spec["highlight_cluster_label"],
            target_pct=spec["highlight_cluster_pct"],
        )

    fig.text(
        0.5,
        0.3,
        f"Doc #{target_id}  ({target_doc['s']}, {n_tokens} tokens)",
        fontsize=13,
        color=FG_MUTED,
        style="italic",
        ha="center",
        va="center",
        family="sans-serif",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=BG_PAGE)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
