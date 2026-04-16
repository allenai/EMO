"""
For each document and layer, sort experts by avg probability (descending),
then compute cumulative probability mass. Plot as scatterplots per layer.
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-file", required=True, help="Path to embeddings_probs.npy")
    parser.add_argument("--info-file", required=True, help="Path to info.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default=None, help="Label for plot titles")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load info
    with open(args.info_file) as f:
        info = json.load(f)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]

    # Load embeddings: (num_docs, num_layers * num_experts)
    emb = np.load(args.emb_file).astype(np.float32)
    num_docs = emb.shape[0]
    print(
        f"Loaded {args.emb_file}: shape={emb.shape}, num_docs={num_docs}, "
        f"num_layers={num_layers}, num_experts={num_experts}"
    )

    # Reshape to (num_docs, num_layers, num_experts)
    emb = emb.reshape(num_docs, num_layers, num_experts)

    # For each doc, each layer: sort experts descending, compute cumulative sum
    # Sort descending along expert axis
    emb_sorted = np.sort(emb, axis=2)[:, :, ::-1]  # (docs, layers, experts) sorted desc

    # Cumulative sum along expert axis
    cumsum = np.cumsum(emb_sorted, axis=2)  # (docs, layers, experts)

    # Normalize so that the last column = 1.0
    total = cumsum[:, :, -1:]  # (docs, layers, 1)
    total = np.where(total == 0, 1.0, total)  # avoid div by zero
    cumsum_norm = cumsum / total  # (docs, layers, experts)

    label = args.model_label or Path(args.emb_file).parent.name

    # Plot: one figure per layer
    # Use 4x4 grid for 16 layers
    fig, axes = plt.subplots(4, 4, figsize=(28, 24), dpi=100)
    fig.suptitle(f"Cumulative Expert Probability Mass (sorted desc)\n{label}", fontsize=16, y=0.98)

    x = np.arange(1, num_experts + 1)

    for layer_idx in range(num_layers):
        ax = axes[layer_idx // 4, layer_idx % 4]
        y_data = cumsum_norm[:, layer_idx, :]  # (num_docs, num_experts)

        # Subsample docs for scatter to avoid overplotting
        max_scatter = 2000
        if num_docs > max_scatter:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(num_docs, max_scatter, replace=False)
            y_sample = y_data[sample_idx]
        else:
            y_sample = y_data

        # Plot scatter — single vectorized call (tile x for all docs)
        x_tiled = np.tile(x, y_sample.shape[0])
        y_flat = y_sample.ravel()
        ax.scatter(
            x_tiled,
            y_flat,
            s=0.3,
            alpha=0.05,
            color="steelblue",
            rasterized=True,
            edgecolors="none",
        )

        # Overlay percentile lines for readability
        for pct, color, lw in [(50, "red", 1.5), (10, "orange", 1.0), (90, "orange", 1.0)]:
            pct_vals = np.percentile(y_data, pct, axis=0)
            ax.plot(x, pct_vals, color=color, lw=lw, label=f"p{pct}" if layer_idx == 0 else None)

        # Mean line
        mean_vals = y_data.mean(axis=0)
        ax.plot(
            x,
            mean_vals,
            color="black",
            lw=1.5,
            linestyle="--",
            label="mean" if layer_idx == 0 else None,
        )

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Top-k experts")
        ax.set_ylabel("Cumulative prob mass")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, min(num_experts + 1, 130))
        ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    # Add legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "cumulative_expert_mass_by_layer.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save a zoomed version focusing on top-32 experts
    fig2, axes2 = plt.subplots(4, 4, figsize=(28, 24), dpi=100)
    fig2.suptitle(f"Cumulative Expert Prob Mass — Zoom to top-32\n{label}", fontsize=16, y=0.98)

    x_zoom = np.arange(1, 33)

    for layer_idx in range(num_layers):
        ax = axes2[layer_idx // 4, layer_idx % 4]
        y_data = cumsum_norm[:, layer_idx, :32]  # (num_docs, 32)

        max_scatter = 2000
        if num_docs > max_scatter:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(num_docs, max_scatter, replace=False)
            y_sample = y_data[sample_idx]
        else:
            y_sample = y_data

        x_tiled = np.tile(x_zoom, y_sample.shape[0])
        y_flat = y_sample.ravel()
        ax.scatter(
            x_tiled,
            y_flat,
            s=0.8,
            alpha=0.05,
            color="steelblue",
            rasterized=True,
            edgecolors="none",
        )

        for pct, color, lw in [(50, "red", 1.5), (10, "orange", 1.0), (90, "orange", 1.0)]:
            pct_vals = np.percentile(y_data, pct, axis=0)
            ax.plot(
                x_zoom, pct_vals, color=color, lw=lw, label=f"p{pct}" if layer_idx == 0 else None
            )

        mean_vals = y_data.mean(axis=0)
        ax.plot(
            x_zoom,
            mean_vals,
            color="black",
            lw=1.5,
            linestyle="--",
            label="mean" if layer_idx == 0 else None,
        )

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Top-k experts")
        ax.set_ylabel("Cumulative prob mass")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, 33)
        ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path2 = output_dir / "cumulative_expert_mass_by_layer_zoom32.png"
    fig2.savefig(out_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
