"""Side-by-side document-probs topic-similarity heatmaps for two MoE models.

Mirrors the topic-topic cosine-similarity panel of
``src/scripts/clustering/plot_doc_expert_coverage.py`` but renders TWO
models in one combined figure: a left 2x2 of similarity heatmaps for the
"reg" (vanilla MoE) model and a right 2x2 for the "modular" (two-level
batch-LB) model. Both subfigures use 4 evenly-spaced layers and a shared
colormap so the panels are directly comparable.

Reads the aggregated ``doc_probs_heatmap_data.npz`` file from each model
dir (produced by the upstream extraction pipeline). The npz contains
``topics`` and ``topic_avg_emb`` of shape (num_topics, num_layers,
num_experts), which is all we need to compute topic-topic similarities.

Output:
    claude_outputs/other_figures/doc_probs_similarity_compare.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "claude_outputs" / "clustering" / "weborganizer"

DEFAULT_LEFT_MODEL = (
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419"
)
DEFAULT_RIGHT_MODEL = (
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419"
)

LEFT_LABEL = "Reg. MoE (1T)"
RIGHT_LABEL = "ModMoE (1T)"

DEFAULT_OUTPUT = (
    REPO_ROOT / "claude_outputs" / "other_figures" / "doc_probs_similarity_compare.png"
)

AGGREGATED_FILENAME = "doc_probs_heatmap_data.npz"

# Colormap chosen to match the paper's pink/magenta theme used in the
# main_results / validation figures while remaining perceptually-uniform.
SIMILARITY_CMAP = "RdPu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Parent directory containing the per-model subdirectories.",
    )
    parser.add_argument("--left-model", default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--right-model", default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--left-label", default=LEFT_LABEL)
    parser.add_argument("--right-label", default=RIGHT_LABEL)
    parser.add_argument(
        "--topic-order-file", type=Path, default=None,
        help="Default: <data-root>/topic_order.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _topic_similarity(vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix from (num_topics, num_experts) vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vecs_n = vecs / norms
    return vecs_n @ vecs_n.T


def _load_model_similarity_matrices(
    model_dir: Path,
    topic_order: List[str],
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Returns (similarity matrices, layer indices, ordered topic names)."""
    npz_path = model_dir / AGGREGATED_FILENAME
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing aggregated data file: {npz_path}\n"
            "Re-run the upstream extraction/aggregation pipeline."
        )

    logger.info(f"Loading {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    file_topics = [str(t) for t in data["topics"]]
    topic_avg = data["topic_avg_emb"]  # (num_topics, num_layers, num_experts)
    num_topics, num_layers, _ = topic_avg.shape

    if set(file_topics) != set(topic_order):
        raise ValueError(
            f"Topic-order topics differ from npz topics for {model_dir}.\n"
            f"  In file but not order: {sorted(set(file_topics) - set(topic_order))}\n"
            f"  In order but not file: {sorted(set(topic_order) - set(file_topics))}"
        )

    # Reorder rows to match topic_order.
    perm = [file_topics.index(t) for t in topic_order]
    topic_avg = topic_avg[perm]

    selected_layers = np.linspace(0, num_layers - 1, 4).astype(int).tolist()
    sim_mats = [
        _topic_similarity(topic_avg[:, layer_idx, :])
        for layer_idx in selected_layers
    ]
    return sim_mats, selected_layers, list(topic_order)


def _draw_quad(
    axes_block,
    sim_mats: List[np.ndarray],
    layer_indices: List[int],
    topics: List[str],
    vmin: float,
    vmax: float,
) -> List["plt.AxesImage"]:
    """Plot 4 similarity heatmaps into a 2x2 block of axes.

    Heatmap cells are square (aspect="equal"). To reduce label repetition
    within the block, y-tick labels appear only on the leftmost column
    and x-tick labels only on the bottom row.
    """
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
            ax.set_xticklabels(topics, fontsize=6, rotation=90)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_yticklabels(topics, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=2)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
    return images


def main() -> None:
    args = parse_args()
    topic_order_file = (
        args.topic_order_file
        if args.topic_order_file is not None
        else args.data_root / "topic_order.json"
    )
    if not topic_order_file.exists():
        raise FileNotFoundError(
            f"Topic-order file not found at {topic_order_file}."
        )
    with open(topic_order_file) as f:
        topic_order = json.load(f)
    logger.info(f"Topic order from {topic_order_file} ({len(topic_order)} topics)")

    left_dir = args.data_root / args.left_model
    right_dir = args.data_root / args.right_model

    left_mats, left_layers, left_topics = _load_model_similarity_matrices(
        left_dir, topic_order
    )
    right_mats, right_layers, right_topics = _load_model_similarity_matrices(
        right_dir, topic_order
    )

    # Shared color range so the two halves are directly comparable.
    all_vals = np.concatenate(
        [m.flatten() for m in left_mats] + [m.flatten() for m in right_mats]
    )
    vmin = max(0.0, float(np.min(all_vals)))
    vmax = min(1.0, float(np.max(all_vals)))

    # 2 rows x (2 + spacer + 2) cols. The spacer column gives the right
    # block's leftmost y-tick labels somewhere to live without overlapping
    # the left block's rightmost panel.
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

    # Group titles centered above each 2x2 block, in figure coords.
    # y chosen to sit just above the per-panel "Layer X" subplot titles.
    fig.text(
        0.27, 0.88, args.left_label,
        ha="center", va="bottom", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.72, 0.88, args.right_label,
        ha="center", va="bottom", fontsize=15, fontweight="bold",
    )

    all_axes = [
        ax for block in (left_block, right_block) for row in block for ax in row
    ]
    cbar = fig.colorbar(
        images[0], ax=all_axes,
        shrink=0.85, pad=0.02, fraction=0.025,
    )
    cbar.set_label("Cosine similarity (topic vs. topic)", fontsize=11)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
