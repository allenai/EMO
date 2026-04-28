"""
Plot expert-coverage heatmaps from a per-document embedding file produced by
extract_document.py (either embeddings_doc_topk_freq.npy or
embeddings_doc_probs.npy).

Produces five PNGs in --output-dir, prefixed by the embedding name:

  <prefix>_coverage_above_uniform_heatmap.png
      topics x layers, value = avg # experts whose weight > 1/num_experts.
      Universal threshold makes the topk_freq and probs plots directly
      comparable.

  <prefix>_coverage_nonzero_heatmap.png
      topics x layers, value = avg # experts with weight > 0. For
      topk_freq this is "any expert ever selected"; for probs it is
      degenerate (=num_experts everywhere) since softmax outputs are
      strictly positive -- emitted for symmetry.

  <prefix>_entropy_heatmap.png
      topics x layers, value = entropy (bits) of the layer's normalized
      expert distribution per doc, averaged. topk_freq is renormalized
      to sum to 1 (divided by routed_top_k) before entropy.

  <prefix>_similarity_heatmap.png
      2x2 panels of topic-topic cosine similarity at 4 evenly-spaced
      layers (np.linspace(0, num_layers-1, 4)).

  <prefix>_l2_distance_heatmap.png
      2x2 panels of topic-topic L2 distance at the same layers.

Topic ordering: pass --topic-order-file <path> to share an ordering across
multiple plot runs (e.g. across both embedding types and across models).
If the file exists, its order is used. Otherwise the order is computed
from THIS embedding's mean entropy (ascending = most concentrated first)
and written to the file so later runs reuse it. With no flag, the order
is derived per-invocation and not persisted.

Usage
-----
    python -u -m src.scripts.clustering.plot_doc_expert_coverage \\
        --emb-file claude_outputs/clustering/weborganizer/<model>/embeddings_doc_topk_freq.npy \\
        --topic-order-file claude_outputs/clustering/weborganizer/topic_order.json

By default, output-dir, metadata-file, and info-file are derived from --emb-file's
parent directory.
"""

import argparse
import gzip
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _derive_prefix(emb_file: str) -> str:
    """embeddings_doc_topk_freq.npy -> doc_topk_freq"""
    base = os.path.basename(emb_file)
    if base.startswith("embeddings_"):
        base = base[len("embeddings_") :]
    if base.endswith(".npy"):
        base = base[:-4]
    return base


def _load_labels(metadata_file: str) -> np.ndarray:
    labels = []
    with gzip.open(metadata_file, "rt") as f:
        for line in f:
            labels.append(json.loads(line)["source"])
    return np.array(labels)


def _per_topic_average(emb_3d: np.ndarray, labels: np.ndarray, topics: list) -> np.ndarray:
    """Returns (num_topics, num_layers, num_experts)."""
    out = []
    for topic in topics:
        mask = labels == topic
        out.append(emb_3d[mask].mean(axis=0))
    return np.stack(out, axis=0)


def _layer_distribution(emb_3d: np.ndarray) -> np.ndarray:
    """Renormalize each layer slice to a probability distribution."""
    sums = emb_3d.sum(axis=-1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return emb_3d / sums


def _per_doc_entropy(dist_3d: np.ndarray) -> np.ndarray:
    """
    Entropy in bits per (doc, layer).
    dist_3d: (num_docs, num_layers, num_experts), each (doc, layer) slice
             sums to 1.
    Returns (num_docs, num_layers).
    """
    num_docs, num_layers, _ = dist_3d.shape
    flat = dist_3d.reshape(num_docs * num_layers, -1)
    H = scipy_entropy(flat.T, base=2)
    return H.reshape(num_docs, num_layers)


# ── Plot helpers ─────────────────────────────────────────────────────────────


def _heatmap_topic_layer(
    matrix: np.ndarray,
    topics: list,
    num_layers: int,
    title: str,
    cbar_label: str,
    out_path: str,
    cmap: str = "YlOrRd",
) -> None:
    fig_h = max(8, len(topics) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)], fontsize=9)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Topic", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {out_path}")


def _heatmap_4panel(
    layer_matrices: list,
    layer_indices: list,
    topics: list,
    suptitle: str,
    out_path: str,
    cmap: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, (mat, layer_idx) in enumerate(zip(layer_matrices, layer_indices)):
        ax = axes[idx // 2][idx % 2]
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, fontsize=7, rotation=90)
        ax.set_yticks(range(len(topics)))
        ax.set_yticklabels(topics, fontsize=7)
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot expert-coverage heatmaps from a per-doc embedding file"
    )
    parser.add_argument("--emb-file", required=True, help="embeddings_doc_*.npy")
    parser.add_argument("--metadata-file", default=None, help="default: <emb-dir>/metadata_docs.jsonl.gz")
    parser.add_argument("--info-file", default=None, help="default: <emb-dir>/info.json")
    parser.add_argument("--output-dir", default=None, help="default: <emb-dir>")
    parser.add_argument(
        "--topic-order-file",
        default=None,
        help=(
            "Shared topic-order JSON file. If exists, its ordering is used. "
            "If not, derived from this embedding's mean entropy and written to "
            "the file (so subsequent runs reuse it). Without this flag, ordering "
            "is per-invocation and not persisted."
        ),
    )
    args = parser.parse_args()

    emb_dir = os.path.dirname(args.emb_file)
    if args.metadata_file is None:
        args.metadata_file = os.path.join(emb_dir, "metadata_docs.jsonl.gz")
    if args.info_file is None:
        args.info_file = os.path.join(emb_dir, "info.json")
    if args.output_dir is None:
        args.output_dir = emb_dir
    os.makedirs(args.output_dir, exist_ok=True)

    prefix = _derive_prefix(args.emb_file)
    logger.info(f"Embedding prefix: {prefix}")

    with open(args.info_file) as f:
        info = json.load(f)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    uniform_threshold = 1.0 / num_experts

    logger.info(f"Loading {args.emb_file} ...")
    emb = np.load(args.emb_file)  # (num_docs, num_layers * num_experts)
    num_docs = emb.shape[0]
    emb_3d = emb.reshape(num_docs, num_layers, num_experts)

    labels = _load_labels(args.metadata_file)
    assert len(labels) == num_docs, f"label count {len(labels)} != num_docs {num_docs}"

    unique_topics = sorted(set(labels.tolist()))
    logger.info(f"Loaded {num_docs} docs across {len(unique_topics)} topics")

    # ── Per-topic stats ──────────────────────────────────────────────────────
    above_uniform_per_doc_layer = (emb_3d > uniform_threshold).sum(axis=2)  # (D, L)
    nonzero_per_doc_layer = (emb_3d > 0).sum(axis=2)  # (D, L)
    dist_3d = _layer_distribution(emb_3d)
    entropy_per_doc_layer = _per_doc_entropy(dist_3d)  # (D, L)

    topic_above_uniform = np.zeros((len(unique_topics), num_layers))
    topic_nonzero = np.zeros((len(unique_topics), num_layers))
    topic_entropy = np.zeros((len(unique_topics), num_layers))
    topic_avg_emb = np.zeros((len(unique_topics), num_layers, num_experts))

    for ti, topic in enumerate(unique_topics):
        mask = labels == topic
        topic_above_uniform[ti] = above_uniform_per_doc_layer[mask].mean(axis=0)
        topic_nonzero[ti] = nonzero_per_doc_layer[mask].mean(axis=0)
        topic_entropy[ti] = entropy_per_doc_layer[mask].mean(axis=0)
        topic_avg_emb[ti] = emb_3d[mask].mean(axis=0)

    # Resolve topic ordering. Either load from a shared file, derive from
    # this embedding's mean entropy and persist, or derive without persisting.
    if args.topic_order_file and os.path.exists(args.topic_order_file):
        with open(args.topic_order_file) as f:
            ordered_topics = json.load(f)
        if set(ordered_topics) != set(unique_topics):
            raise ValueError(
                f"--topic-order-file {args.topic_order_file} topics "
                f"{sorted(ordered_topics)} != extracted topics "
                f"{sorted(unique_topics)}"
            )
        order = np.array([unique_topics.index(t) for t in ordered_topics])
        logger.info(f"Loaded topic order from {args.topic_order_file}")
    else:
        order = np.argsort(topic_entropy.mean(axis=1))
        if args.topic_order_file:
            ordered_topics = [unique_topics[i] for i in order]
            os.makedirs(os.path.dirname(args.topic_order_file) or ".", exist_ok=True)
            with open(args.topic_order_file, "w") as f:
                json.dump(ordered_topics, f, indent=2)
            logger.info(f"Wrote topic order -> {args.topic_order_file}")
    topics = [unique_topics[i] for i in order]
    topic_above_uniform = topic_above_uniform[order]
    topic_nonzero = topic_nonzero[order]
    topic_entropy = topic_entropy[order]
    topic_avg_emb = topic_avg_emb[order]

    max_entropy = float(np.log2(num_experts))

    # ── Heatmap 1: above-uniform coverage ────────────────────────────────────
    logger.info("\nPlotting coverage (above-uniform threshold) heatmap ...")
    _heatmap_topic_layer(
        topic_above_uniform,
        topics,
        num_layers,
        title=f"Avg # experts with weight > 1/{num_experts}  ({prefix})",
        cbar_label=f"# experts above uniform (out of {num_experts})",
        out_path=os.path.join(
            args.output_dir, f"{prefix}_coverage_above_uniform_heatmap.png"
        ),
    )

    # ── Heatmap 2: nonzero coverage ──────────────────────────────────────────
    logger.info("Plotting coverage (nonzero) heatmap ...")
    _heatmap_topic_layer(
        topic_nonzero,
        topics,
        num_layers,
        title=f"Avg # experts with weight > 0  ({prefix})",
        cbar_label=f"# experts nonzero (out of {num_experts})",
        out_path=os.path.join(args.output_dir, f"{prefix}_coverage_nonzero_heatmap.png"),
    )

    # ── Heatmap 3: entropy ───────────────────────────────────────────────────
    logger.info("Plotting entropy heatmap ...")
    _heatmap_topic_layer(
        topic_entropy,
        topics,
        num_layers,
        title=f"Expert-selection entropy per layer  ({prefix})",
        cbar_label=f"Entropy (bits, max={max_entropy:.2f})",
        out_path=os.path.join(args.output_dir, f"{prefix}_entropy_heatmap.png"),
    )

    # ── Heatmap 4 & 5: topic-topic similarity / L2 at 4 layers ──────────────
    selected_layers = np.linspace(0, num_layers - 1, 4).astype(int).tolist()
    logger.info(f"Plotting topic-topic similarity / L2 at layers {selected_layers} ...")

    sim_mats = []
    l2_mats = []
    for layer_idx in selected_layers:
        vecs = topic_avg_emb[:, layer_idx, :]  # (num_topics, num_experts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        vecs_n = vecs / norms
        sim_mats.append(vecs_n @ vecs_n.T)

        sq = (vecs**2).sum(axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2 * (vecs @ vecs.T)
        l2_mats.append(np.sqrt(np.maximum(dist_sq, 0)))

    _heatmap_4panel(
        sim_mats,
        selected_layers,
        topics,
        suptitle=f"Topic-Topic Cosine Similarity ({prefix})",
        out_path=os.path.join(args.output_dir, f"{prefix}_similarity_heatmap.png"),
        cmap="YlOrRd",
    )
    _heatmap_4panel(
        l2_mats,
        selected_layers,
        topics,
        suptitle=f"Topic-Topic L2 Distance ({prefix})",
        out_path=os.path.join(args.output_dir, f"{prefix}_l2_distance_heatmap.png"),
        cmap="YlOrRd_r",
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
