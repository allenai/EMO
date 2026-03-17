"""
Plot expert coverage heatmaps from topic_stats.json and expert_freq.npy.

Generates:
  1. Heatmap: topics x layers x avg experts used
  2. Heatmap: topics x layers x entropy
  3. 2x2 topic-topic dot product similarity at layers L0, L5, L10, L15

Usage:
    python -u -m src.scripts.analysis.plot_expert_coverage \
        --stats-file claude_outputs/analysis/expert_coverage_weborganizer/topic_stats.json \
        --output-dir claude_outputs/analysis/expert_coverage_weborganizer
"""

import argparse
import gzip
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Plot expert coverage heatmaps")
    parser.add_argument("--stats-file", type=str, required=True,
                        help="Path to topic_stats.json")
    parser.add_argument("--freq-file", type=str, default=None,
                        help="Path to expert_freq.npy (defaults to same dir as stats-file)")
    parser.add_argument("--metadata-file", type=str, default=None,
                        help="Path to metadata.jsonl.gz (defaults to same dir as stats-file)")
    parser.add_argument("--info-file", type=str, default=None,
                        help="Path to info.json (defaults to same dir as stats-file)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to same dir as stats-file)")
    args = parser.parse_args()

    stats_dir = os.path.dirname(args.stats_file)
    if args.output_dir is None:
        args.output_dir = stats_dir
    if args.freq_file is None:
        args.freq_file = os.path.join(stats_dir, "expert_freq.npy")
    if args.metadata_file is None:
        args.metadata_file = os.path.join(stats_dir, "metadata.jsonl.gz")
    if args.info_file is None:
        args.info_file = os.path.join(stats_dir, "info.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.stats_file) as f:
        topic_stats = json.load(f)

    # Sort topics by mean entropy (most concentrated first)
    topics = sorted(topic_stats.keys(),
                    key=lambda t: topic_stats[t]["entropy_per_layer_mean"])
    num_topics = len(topics)
    num_layers = len(topic_stats[topics[0]]["avg_experts_per_layer"])

    max_entropy = topic_stats[topics[0]]["max_entropy"]
    fig_h = max(8, num_topics * 0.45)

    # ── Heatmap 1: Average experts used per layer ────────────────────────────
    matrix_experts = np.array([topic_stats[t]["avg_experts_per_layer"] for t in topics])

    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(matrix_experts, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)], fontsize=9)
    ax.set_yticks(range(num_topics))
    ax.set_yticklabels(topics, fontsize=9)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Topic", fontsize=11)
    ax.set_title("Average Experts Used per Layer per Topic", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Avg experts used (out of 127)", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "expert_coverage_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap -> {out_path}")

    # ── Heatmap 2: Entropy per layer ─────────────────────────────────────────
    matrix_entropy = np.array([topic_stats[t]["entropy_per_layer"] for t in topics])

    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(matrix_entropy, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)], fontsize=9)
    ax.set_yticks(range(num_topics))
    ax.set_yticklabels(topics, fontsize=9)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Topic", fontsize=11)
    ax.set_title("Expert Selection Entropy per Layer per Topic", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Entropy (bits, max={max_entropy:.2f})", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "expert_entropy_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap -> {out_path}")

    # ── Heatmap 3: Topic-topic similarity at selected layers ─────────────────
    if not os.path.exists(args.freq_file):
        logger.warning(f"Skipping similarity plot: {args.freq_file} not found")
        return

    logger.info("Loading frequency data for similarity plot ...")
    all_freq = np.load(args.freq_file)  # (num_docs, num_layers * num_experts)
    with open(args.info_file) as f:
        info = json.load(f)
    num_experts = info["num_standard_experts"]

    # Load per-doc labels
    labels = []
    with gzip.open(args.metadata_file, "rt") as f:
        for line in f:
            labels.append(json.loads(line)["source"])
    labels = np.array(labels)

    # Reshape to (num_docs, num_layers, num_experts)
    all_freq_3d = all_freq.reshape(len(labels), num_layers, num_experts)

    # Compute per-topic average frequency vector at each layer
    # topic_avg[topic] = (num_layers, num_experts)
    topic_avg = {}
    for topic in topics:
        mask = labels == topic
        topic_avg[topic] = all_freq_3d[mask].mean(axis=0)

    selected_layers = [0, 5, 10, 15]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for idx, layer_idx in enumerate(selected_layers):
        ax = axes[idx // 2][idx % 2]

        # Build (num_topics, num_experts) matrix for this layer
        vecs = np.array([topic_avg[t][layer_idx] for t in topics])

        # Cosine similarity: normalize rows then dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_normed = vecs / norms
        sim = vecs_normed @ vecs_normed.T

        im = ax.imshow(sim, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(num_topics))
        ax.set_xticklabels(topics, fontsize=7, rotation=90)
        ax.set_yticks(range(num_topics))
        ax.set_yticklabels(topics, fontsize=7)
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle("Topic-Topic Cosine Similarity (avg freq vectors per layer)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "topic_similarity_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap -> {out_path}")

    # ── Heatmap 4: Topic-topic L2 distance at selected layers ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for idx, layer_idx in enumerate(selected_layers):
        ax = axes[idx // 2][idx % 2]

        vecs = np.array([topic_avg[t][layer_idx] for t in topics])

        # Pairwise L2 distance
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq_norms = (vecs ** 2).sum(axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (vecs @ vecs.T)
        dist = np.sqrt(np.maximum(dist_sq, 0))

        im = ax.imshow(dist, aspect="auto", cmap="YlOrRd_r")
        ax.set_xticks(range(num_topics))
        ax.set_xticklabels(topics, fontsize=7, rotation=90)
        ax.set_yticks(range(num_topics))
        ax.set_yticklabels(topics, fontsize=7)
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle("Topic-Topic L2 Distance (avg freq vectors per layer)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "topic_l2_distance_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap -> {out_path}")


if __name__ == "__main__":
    main()
