"""V2 of plot_doc_probs_similarity_compare.py — top-N topics + bigger labels.

Differences from v1:
    - Restrict the similarity heatmaps to the first ``--top-n`` topics
      (default 10) from topic_order.json so each tile gets a lot more
      label real estate.
    - Topic labels rendered at a larger font size, and x-axis labels
      slanted (45° + right-anchored) instead of vertical for legibility.

Output:
    claude_outputs/other_figures/doc_probs_similarity_compare_v2.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from plot_doc_probs_similarity_compare import (
    AGGREGATED_FILENAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_LEFT_MODEL,
    DEFAULT_RIGHT_MODEL,
    LEFT_LABEL,
    RIGHT_LABEL,
    SIMILARITY_CMAP,
    _topic_similarity,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "claude_outputs" / "other_figures"
    / "doc_probs_similarity_compare_v2.pdf"
)

DEFAULT_TOP_N = 10
LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 16
GROUP_TITLE_FONTSIZE = 22
CBAR_LABEL_FONTSIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--left-model", default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--right-model", default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--left-label", default=LEFT_LABEL)
    parser.add_argument("--right-label", default=RIGHT_LABEL)
    parser.add_argument(
        "--topic-order-file", type=Path, default=None,
        help="Default: <data-root>/topic_order.json",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Number of topics (from the start of topic_order.json) to keep.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_model_similarity_matrices_topn(
    model_dir: Path,
    topic_order: List[str],
    top_n: int,
):
    npz_path = model_dir / AGGREGATED_FILENAME
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing aggregated data file: {npz_path}")

    logger.info(f"Loading {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    file_topics = [str(t) for t in data["topics"]]
    topic_avg = data["topic_avg_emb"]
    _, num_layers, _ = topic_avg.shape

    if set(file_topics) != set(topic_order):
        raise ValueError(
            f"Topic-order topics differ from npz topics for {model_dir}.\n"
            f"  In file but not order: {sorted(set(file_topics) - set(topic_order))}\n"
            f"  In order but not file: {sorted(set(topic_order) - set(file_topics))}"
        )

    perm = [file_topics.index(t) for t in topic_order]
    topic_avg = topic_avg[perm][:top_n]
    kept_topics = list(topic_order[:top_n])

    selected_layers = np.linspace(0, num_layers - 1, 4).astype(int).tolist()
    sim_mats = [
        _topic_similarity(topic_avg[:, layer_idx, :])
        for layer_idx in selected_layers
    ]
    return sim_mats, selected_layers, kept_topics


def _draw_quad(axes_block, sim_mats, layer_indices, topics, vmin, vmax):
    images = []
    for idx, (mat, layer_idx) in enumerate(zip(sim_mats, layer_indices)):
        row, col = idx // 2, idx % 2
        ax = axes_block[row][col]
        im = ax.imshow(
            mat, aspect="equal", cmap=SIMILARITY_CMAP, vmin=vmin, vmax=vmax,
        )
        images.append(im)
        ax.set_xticks(range(len(topics)))
        ax.set_yticks(range(len(topics)))
        if row == 1:
            ax.set_xticklabels(
                topics, fontsize=LABEL_FONTSIZE, rotation=45,
                ha="right", rotation_mode="anchor",
            )
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_yticklabels(topics, fontsize=LABEL_FONTSIZE)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=2)
        ax.set_title(f"Layer {layer_idx}", fontsize=TITLE_FONTSIZE)
    return images


def main() -> None:
    args = parse_args()
    topic_order_file = (
        args.topic_order_file
        if args.topic_order_file is not None
        else args.data_root / "topic_order.json"
    )
    if not topic_order_file.exists():
        raise FileNotFoundError(f"Topic-order file not found at {topic_order_file}.")
    with open(topic_order_file) as f:
        topic_order = json.load(f)
    logger.info(
        f"Topic order from {topic_order_file} "
        f"({len(topic_order)} topics; using first {args.top_n})"
    )

    left_dir = args.data_root / args.left_model
    right_dir = args.data_root / args.right_model

    left_mats, left_layers, left_topics = _load_model_similarity_matrices_topn(
        left_dir, topic_order, args.top_n
    )
    right_mats, right_layers, right_topics = _load_model_similarity_matrices_topn(
        right_dir, topic_order, args.top_n
    )

    all_vals = np.concatenate(
        [m.flatten() for m in left_mats] + [m.flatten() for m in right_mats]
    )
    vmin = max(0.0, float(np.min(all_vals)))
    vmax = min(1.0, float(np.max(all_vals)))

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(
        2, 5, width_ratios=[1.0, 1.0, 0.30, 1.0, 1.0],
        wspace=0.06, hspace=-0.08,
    )
    left_block = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ]
    right_block = [
        [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4])],
        [fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[1, 4])],
    ]

    images = []
    images += _draw_quad(left_block, left_mats, left_layers, left_topics, vmin, vmax)
    images += _draw_quad(right_block, right_mats, right_layers, right_topics, vmin, vmax)

    fig.text(
        0.27, 0.88, args.left_label,
        ha="center", va="bottom", fontsize=GROUP_TITLE_FONTSIZE, fontweight="bold",
    )
    fig.text(
        0.72, 0.88, args.right_label,
        ha="center", va="bottom", fontsize=GROUP_TITLE_FONTSIZE, fontweight="bold",
    )

    all_axes = [
        ax for block in (left_block, right_block) for row in block for ax in row
    ]
    cbar = fig.colorbar(
        images[0], ax=all_axes,
        shrink=0.85, pad=0.02, fraction=0.025,
    )
    cbar.set_label("Cosine similarity (topic vs. topic)", fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=CBAR_LABEL_FONTSIZE - 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
