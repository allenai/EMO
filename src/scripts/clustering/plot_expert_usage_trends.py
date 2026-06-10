"""
Cross-model expert-usage trend curves.

Given N extraction dirs (extract_document.py outputs) for models that differ
in total expert count, plot per-topic usage metrics *against expert count*:

  - effective experts per doc (2^entropy of the per-layer expert
    distribution), absolute and as a fraction of the standard-expert count
  - coverage above uniform (# experts with weight > 1/num_standard_experts)

Answers: as the expert pool grows, does a topic use a constant *number* of
experts (specialization deepens elsewhere) or a constant *fraction* (work is
subdivided proportionally)?

Outputs into --output-dir, prefixed by embedding type:
  <emb>_eff_experts_vs_E.png    2 panels (absolute / fraction); one line per
                                layer + bold mean, averaged over topics
  <emb>_coverage_vs_E.png       same layout for coverage-above-uniform
  <emb>_per_topic_eff.png       small-multiple grid, one panel per topic
                                (layer-averaged), absolute + fraction axes
  <emb>_trends.json             full (model, topic, layer) tensors

Usage
-----
    python -u -m src.scripts.clustering.plot_expert_usage_trends \\
        --model-dirs claude_outputs/clustering/weborganizer/emo_1b{4,7,11,14}b_130b \\
        --output-dir claude_outputs/clustering/sizescaling/trends
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as scipy_entropy

from .doc_embeddings import (
    EMB_TYPES,
    layer_distributions,
    load_doc_embeddings,
    load_doc_labels,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_model_stats(data_dir: str, emb_type: str) -> dict:
    """Per-(topic, layer) effective-expert and coverage stats for one model."""
    emb, info = load_doc_embeddings(data_dir, emb_type)
    labels = load_doc_labels(data_dir)
    topics = info["topics"]
    num_experts = info["num_standard_experts"]

    dist = layer_distributions(emb)
    num_docs, num_layers, _ = dist.shape
    H = scipy_entropy(dist.reshape(num_docs * num_layers, -1).T, base=2)
    eff = np.exp2(H).reshape(num_docs, num_layers)  # (D, L)
    cov = (emb > 1.0 / num_experts).sum(axis=-1)  # (D, L)

    eff_tl = np.stack([eff[labels == t].mean(axis=0) for t in topics])  # (T, L)
    cov_tl = np.stack([cov[labels == t].mean(axis=0) for t in topics])  # (T, L)
    return {
        "topics": topics,
        "num_layers": num_layers,
        "num_standard_experts": num_experts,
        "total_experts": num_experts + info["num_shared_experts"],
        "eff_topic_layer": eff_tl,
        "cov_topic_layer": cov_tl,
    }


def _trend_figure(
    stats: list, labels: list, key: str, ylabel: str, title: str, out_path: str
) -> None:
    """2-panel (absolute, fraction-of-experts) trend vs expert count."""
    xs = [s["total_experts"] for s in stats]
    num_layers = stats[0]["num_layers"]
    per_layer = np.stack([s[key].mean(axis=0) for s in stats])  # (M, L), topic-averaged
    denom = np.array([s["num_standard_experts"] for s in stats], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    for panel, (ax, values, ylab) in enumerate(
        zip(axes, [per_layer, per_layer / denom[:, None]], [ylabel, f"{ylabel} / num experts"])
    ):
        for layer in range(num_layers):
            ax.plot(
                xs,
                values[:, layer],
                color=cmap(layer / max(num_layers - 1, 1)),
                alpha=0.55,
                lw=1.2,
            )
        ax.plot(xs, values.mean(axis=1), color="black", lw=2.8, marker="o", label="layer mean")
        ax.set_xlabel("Total experts")
        ax.set_ylabel(ylab)
        ax.set_xticks(xs)
        ax.legend()
        ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_layers - 1))
    fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02, label="Layer")
    fig.suptitle(f"{title}  ({' / '.join(labels)})", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {out_path}")


def _per_topic_figure(stats: list, labels: list, out_path: str) -> None:
    xs = [s["total_experts"] for s in stats]
    topics = stats[0]["topics"]
    denom = np.array([s["num_standard_experts"] for s in stats], dtype=np.float64)
    eff = np.stack([s["eff_topic_layer"].mean(axis=1) for s in stats])  # (M, T), layer-averaged

    ncols = 6
    nrows = int(np.ceil(len(topics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True)
    for ti, topic in enumerate(topics):
        ax = axes.flat[ti]
        ax.plot(xs, eff[:, ti], marker="o", color="tab:blue", lw=1.8)
        ax.set_title(topic, fontsize=9)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, eff[:, ti] / denom, marker="s", color="tab:orange", lw=1.2, alpha=0.7)
        ax2.tick_params(labelsize=7, colors="tab:orange")
    for ti in range(len(topics), nrows * ncols):
        axes.flat[ti].axis("off")
    fig.suptitle(
        "Effective experts per topic vs total experts "
        "(blue: absolute, orange: fraction; layer-averaged)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument(
        "--labels", nargs="+", default=None, help="default: basename of each model dir"
    )
    parser.add_argument("--emb-types", nargs="+", default=list(EMB_TYPES), choices=EMB_TYPES)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.model_dirs]
    assert len(labels) == len(args.model_dirs)
    os.makedirs(args.output_dir, exist_ok=True)

    for emb_type in args.emb_types:
        logger.info(f"=== {emb_type} ===")
        stats = []
        for d, lab in zip(args.model_dirs, labels):
            logger.info(f"Loading {lab} ({d}) ...")
            stats.append(compute_model_stats(d, emb_type))
        order = np.argsort([s["total_experts"] for s in stats])
        stats = [stats[i] for i in order]
        slabels = [labels[i] for i in order]
        assert all(s["topics"] == stats[0]["topics"] for s in stats), "topic sets differ"

        _trend_figure(
            stats,
            slabels,
            "eff_topic_layer",
            "Effective experts (2^H)",
            f"Effective experts per doc vs total experts [{emb_type}]",
            os.path.join(args.output_dir, f"{emb_type}_eff_experts_vs_E.png"),
        )
        _trend_figure(
            stats,
            slabels,
            "cov_topic_layer",
            "Experts above uniform",
            f"Coverage above uniform vs total experts [{emb_type}]",
            os.path.join(args.output_dir, f"{emb_type}_coverage_vs_E.png"),
        )
        _per_topic_figure(
            stats, slabels, os.path.join(args.output_dir, f"{emb_type}_per_topic_eff.png")
        )

        out_json = os.path.join(args.output_dir, f"{emb_type}_trends.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    lab: {
                        "total_experts": s["total_experts"],
                        "num_standard_experts": s["num_standard_experts"],
                        "topics": s["topics"],
                        "eff_topic_layer": s["eff_topic_layer"].tolist(),
                        "cov_topic_layer": s["cov_topic_layer"].tolist(),
                    }
                    for lab, s in zip(slabels, stats)
                },
                f,
                indent=2,
            )
        logger.info(f"  -> {out_json}")


if __name__ == "__main__":
    main()
