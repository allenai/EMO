"""
Per-expert topic profiles and specialization-score distributions.

For each model (extraction dir from extract_document.py), build each expert's
topic profile: p[l, e, t] = (mean usage of expert e at layer l on docs of
topic t), normalized over t — i.e. "given this expert fires, what is it
reading?". The specialization score is the normalized entropy of that
profile: 0 = fires on a single topic, 1 = indiscriminate generalist. The
normalizer log2(num_topics) is constant across models, so scores are directly
comparable between models with different expert counts.

Per model dir, caches:  <model_dir>/expert_profiles_<emb>.npz
    profiles      (num_layers, num_experts, num_topics)
    entropy_norm  (num_layers, num_experts)
    max_share     (num_layers, num_experts)   peak topic share per expert
    topics        (num_topics,) str

Cross-model plots + summary into --output-dir:
    <emb>_entropy_cdf.png        2x2 panels (4 evenly-spaced layers), CDF of
                                 entropy_norm across experts, 1 curve/model
    <emb>_entropy_vs_layer.png   median + IQR band per layer, 1 line/model
    <emb>_max_share_cdf.png      same panels for the peak-topic share
    <emb>_profiles_summary.json  per-model per-layer aggregates

Usage
-----
    python -u -m src.scripts.clustering.expert_topic_profiles \\
        --model-dirs claude_outputs/models_sizescaling/weborganizer/emo_1b{4,7,11,14}b_130b \\
        --output-dir claude_outputs/models_sizescaling/profiles
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

from .doc_embeddings import EMB_TYPES, load_doc_embeddings, load_doc_labels

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_profiles(data_dir: str, emb_type: str, recompute: bool = False) -> dict:
    """Compute (or load cached) per-expert topic profiles for one model."""
    cache = os.path.join(data_dir, f"expert_profiles_{emb_type}.npz")
    if os.path.exists(cache) and not recompute:
        logger.info(f"Loading cached {cache}")
        npz = np.load(cache, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    emb, info = load_doc_embeddings(data_dir, emb_type)
    labels = load_doc_labels(data_dir)
    topics = info["topics"]

    # (T, L, E): mean usage of each expert on each topic's docs
    usage = np.stack([emb[labels == t].mean(axis=0) for t in topics])
    # (L, E, T): normalize over topics -> "which topics does this expert serve"
    usage = usage.transpose(1, 2, 0)
    sums = usage.sum(axis=-1, keepdims=True)
    profiles = usage / np.where(sums > 0, sums, 1.0)

    num_layers, num_experts, num_topics = profiles.shape
    H = scipy_entropy(profiles.reshape(-1, num_topics).T, base=2)
    entropy_norm = (H / np.log2(num_topics)).reshape(num_layers, num_experts)
    max_share = profiles.max(axis=-1)

    out = {
        "profiles": profiles.astype(np.float32),
        "entropy_norm": entropy_norm.astype(np.float32),
        "max_share": max_share.astype(np.float32),
        "topics": np.array(topics),
        "total_experts": np.int64(info["num_standard_experts"] + info["num_shared_experts"]),
    }
    np.savez_compressed(cache, **out)
    logger.info(f"  -> {cache}  profiles shape={profiles.shape}")
    return out


def _cdf_panels(results: list, labels: list, key: str, xlabel: str, out_path: str) -> None:
    num_layers = results[0]["entropy_norm"].shape[0]
    panel_layers = np.unique(np.linspace(0, num_layers - 1, 4).astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, layer in zip(axes.flat, panel_layers):
        for res, lab in zip(results, labels):
            vals = np.sort(res[key][layer])
            ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), lw=1.8, label=lab)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF over experts")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{xlabel}: distribution across experts", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {out_path}")


def _vs_layer_figure(results: list, labels: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for res, lab in zip(results, labels):
        e = res["entropy_norm"]  # (L, E)
        layers = np.arange(e.shape[0])
        med = np.median(e, axis=1)
        q1, q3 = np.percentile(e, [25, 75], axis=1)
        (line,) = ax.plot(layers, med, marker="o", lw=1.8, label=lab)
        ax.fill_between(layers, q1, q3, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized topic entropy (median ± IQR)")
    ax.set_title("Per-expert specialization by layer (lower = more specialized)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--emb-types", nargs="+", default=list(EMB_TYPES), choices=EMB_TYPES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recompute", action="store_true", help="ignore cached npz files")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.model_dirs]
    assert len(labels) == len(args.model_dirs)
    os.makedirs(args.output_dir, exist_ok=True)

    for emb_type in args.emb_types:
        logger.info(f"=== {emb_type} ===")
        results = []
        for d, lab in zip(args.model_dirs, labels):
            logger.info(f"Profiles for {lab} ({d}) ...")
            results.append(compute_profiles(d, emb_type, recompute=args.recompute))
        order = np.argsort([int(r["total_experts"]) for r in results])
        results = [results[i] for i in order]
        slabels = [labels[i] for i in order]

        _cdf_panels(
            results,
            slabels,
            "entropy_norm",
            "Normalized topic entropy",
            os.path.join(args.output_dir, f"{emb_type}_entropy_cdf.png"),
        )
        _cdf_panels(
            results,
            slabels,
            "max_share",
            "Peak topic share",
            os.path.join(args.output_dir, f"{emb_type}_max_share_cdf.png"),
        )
        _vs_layer_figure(
            results, slabels, os.path.join(args.output_dir, f"{emb_type}_entropy_vs_layer.png")
        )

        summary = {}
        for res, lab in zip(results, slabels):
            e = res["entropy_norm"]
            summary[lab] = {
                "total_experts": int(res["total_experts"]),
                "median_entropy_per_layer": np.median(e, axis=1).round(4).tolist(),
                "mean_entropy": float(e.mean()),
                "frac_specialists_lt_0.5": float((e < 0.5).mean()),
            }
        out_json = os.path.join(args.output_dir, f"{emb_type}_profiles_summary.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  -> {out_json}")


if __name__ == "__main__":
    main()
