"""
Cross-model expert matching on per-document usage fingerprints.

Models trained with different expert counts share no initialization, so
expert indices mean nothing across models. But when two extractions
(extract_document.py) ran over the IDENTICAL document set, each expert has a
fingerprint — its usage across all docs — and correspondence can be
discovered functionally: corr[i, j] = Pearson correlation (over docs) between
expert i of model A and expert j of model B at the same layer.

Statistics per layer (A = smaller model by convention):
  matched similarity   Hungarian one-to-one assignment on the correlation
                       matrix — how much of A's organization survives in B
  splitting            per A-expert: # B-experts with corr > tau-match, and
                       the mean pairwise corr among them (low = they
                       partition A's docs, high = redundant copies)
  novelty              per B-expert: max corr to any A expert; "novel" if
                       below tau-novel (no counterpart in A)
  redundancy           within-model nearest-neighbor corr (computed for both
                       models independently)

Outputs into --output-dir:
  corr_heatmap_layers.png      4 evenly-spaced layers, B columns sorted by
                               their Hungarian match
  matched_sim_vs_layer.png     median + IQR of matched corr per layer
  splitting_novelty_vs_layer.png
  nn_redundancy_vs_layer.png
  corr_matrices.npz            (num_layers, E_A, E_B) float32 + per-layer stats
  match_summary.json

Usage
-----
    python -u -m src.scripts.clustering.match_experts \\
        --dir-a claude_outputs/models_sizescaling/weborganizer/emo_1b4b_130b \\
        --dir-b claude_outputs/models_sizescaling/weborganizer/emo_1b14b_130b \\
        --output-dir claude_outputs/models_sizescaling/matching/1b4b_vs_1b14b
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from .doc_embeddings import EMB_TYPES, assert_same_doc_set, load_doc_embeddings

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    """Center + unit-norm each column; zero-variance columns become all-zero."""
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(x, axis=0, keepdims=True)
    return x / np.where(norm > 1e-12, norm, 1.0)


def layer_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation over docs between expert columns: (E_A, E_B)."""
    return _standardize_columns(a).T @ _standardize_columns(b)


def nn_offdiag(corr_self: np.ndarray) -> np.ndarray:
    """Per-expert nearest-neighbor corr within one model (diagonal excluded)."""
    c = corr_self.copy()
    np.fill_diagonal(c, -np.inf)
    return c.max(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir-a", required=True, help="smaller model's extraction dir")
    parser.add_argument("--dir-b", required=True, help="larger model's extraction dir")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--emb-type", default="probs", choices=EMB_TYPES)
    parser.add_argument("--tau-match", type=float, default=0.4)
    parser.add_argument("--tau-novel", type=float, default=0.3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    label_a = args.label_a or os.path.basename(os.path.normpath(args.dir_a))
    label_b = args.label_b or os.path.basename(os.path.normpath(args.dir_b))
    os.makedirs(args.output_dir, exist_ok=True)

    assert_same_doc_set(args.dir_a, args.dir_b)
    emb_a, info_a = load_doc_embeddings(args.dir_a, args.emb_type)
    emb_b, info_b = load_doc_embeddings(args.dir_b, args.emb_type)
    assert info_a["num_layers"] == info_b["num_layers"]
    num_layers = info_a["num_layers"]
    e_a, e_b = emb_a.shape[2], emb_b.shape[2]
    logger.info(f"{label_a}: {e_a} experts, {label_b}: {e_b} experts, {num_layers} layers")

    corr_all = np.zeros((num_layers, e_a, e_b), dtype=np.float32)
    per_layer = []
    for layer in range(num_layers):
        a, b = emb_a[:, layer, :], emb_b[:, layer, :]
        corr = layer_correlations(a, b)
        corr_all[layer] = corr

        rows, cols = linear_sum_assignment(-corr)
        matched = corr[rows, cols]

        corr_bb = layer_correlations(b, b)
        splits, coherences = [], []
        for i in range(e_a):
            js = np.where(corr[i] > args.tau_match)[0]
            splits.append(len(js))
            if len(js) >= 2:
                sub = corr_bb[np.ix_(js, js)]
                coherences.append(float(sub[np.triu_indices(len(js), k=1)].mean()))
        novelty = corr.max(axis=0)  # per B-expert best match to A

        per_layer.append(
            {
                "matched_sim": matched,
                "hungarian_cols": cols,
                "splits": np.array(splits),
                "split_coherence": float(np.mean(coherences)) if coherences else float("nan"),
                "novelty_max_corr": novelty,
                "frac_novel_b": float((novelty < args.tau_novel).mean()),
                "nn_a": nn_offdiag(layer_correlations(a, a)),
                "nn_b": nn_offdiag(corr_bb),
            }
        )
        logger.info(
            f"  L{layer:02d}: matched corr med={np.median(matched):.3f}, "
            f"splits/A-expert mean={np.mean(splits):.2f}, "
            f"novel B frac={per_layer[-1]['frac_novel_b']:.2f}"
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    panel_layers = np.unique(np.linspace(0, num_layers - 1, 4).astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, layer in zip(axes.flat, panel_layers):
        cols = per_layer[layer]["hungarian_cols"]
        rest = np.setdiff1d(np.arange(e_b), cols)
        im = ax.imshow(
            corr_all[layer][:, np.concatenate([cols, rest])],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Layer {layer} (first {e_a} cols = Hungarian match)", fontsize=10)
        ax.set_xlabel(f"{label_b} experts (sorted)")
        ax.set_ylabel(f"{label_a} experts")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Expert fingerprint correlation [{args.emb_type}]", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "corr_heatmap_layers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {path}")

    layers = np.arange(num_layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    med = [np.median(pl["matched_sim"]) for pl in per_layer]
    q1 = [np.percentile(pl["matched_sim"], 25) for pl in per_layer]
    q3 = [np.percentile(pl["matched_sim"], 75) for pl in per_layer]
    ax.plot(layers, med, marker="o", color="tab:blue")
    ax.fill_between(layers, q1, q3, alpha=0.2, color="tab:blue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Hungarian-matched corr (median ± IQR)")
    ax.set_title(f"{label_a} ↔ {label_b}: one-to-one matched similarity")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(alpha=0.3)
    path = os.path.join(args.output_dir, "matched_sim_vs_layer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers,
        [pl["splits"].mean() for pl in per_layer],
        marker="o",
        color="tab:green",
        label=f"mean # {label_b}-experts matching one {label_a}-expert (corr>{args.tau_match})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean matches per A-expert")
    ax2 = ax.twinx()
    ax2.plot(
        layers,
        [pl["frac_novel_b"] for pl in per_layer],
        marker="s",
        color="tab:red",
        label=f"frac {label_b}-experts novel (max corr<{args.tau_novel})",
    )
    ax2.set_ylabel("Fraction novel", color="tab:red")
    ax2.set_ylim(0, 1)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper center")
    ax.set_title(f"{label_a} ↔ {label_b}: splitting and novelty")
    ax.grid(alpha=0.3)
    path = os.path.join(args.output_dir, "splitting_novelty_vs_layer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, lab, color in [("nn_a", label_a, "tab:blue"), ("nn_b", label_b, "tab:orange")]:
        med = [np.median(pl[key]) for pl in per_layer]
        q1 = [np.percentile(pl[key], 25) for pl in per_layer]
        q3 = [np.percentile(pl[key], 75) for pl in per_layer]
        ax.plot(layers, med, marker="o", color=color, label=lab)
        ax.fill_between(layers, q1, q3, alpha=0.15, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Within-model NN corr (median ± IQR)")
    ax.set_title("Within-model expert redundancy")
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(args.output_dir, "nn_redundancy_vs_layer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {path}")

    # ── Save raw + summary ───────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(args.output_dir, "corr_matrices.npz"),
        corr=corr_all,
        matched_sim=np.stack([pl["matched_sim"] for pl in per_layer]),
        splits=np.stack([pl["splits"] for pl in per_layer]),
        novelty_max_corr=np.stack([pl["novelty_max_corr"] for pl in per_layer]),
        nn_a=np.stack([pl["nn_a"] for pl in per_layer]),
        nn_b=np.stack([pl["nn_b"] for pl in per_layer]),
    )
    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "emb_type": args.emb_type,
        "tau_match": args.tau_match,
        "tau_novel": args.tau_novel,
        "num_experts_a": e_a,
        "num_experts_b": e_b,
        "per_layer": [
            {
                "layer": layer,
                "matched_sim_median": float(np.median(pl["matched_sim"])),
                "mean_splits_per_a_expert": float(pl["splits"].mean()),
                "split_coherence": pl["split_coherence"],
                "frac_novel_b": pl["frac_novel_b"],
                "nn_a_median": float(np.median(pl["nn_a"])),
                "nn_b_median": float(np.median(pl["nn_b"])),
            }
            for layer, pl in enumerate(per_layer)
        ],
    }
    path = os.path.join(args.output_dir, "match_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  -> {path}")


if __name__ == "__main__":
    main()
