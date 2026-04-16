"""
Apply temperature scaling to averaged router logits, re-softmax,
then plot cumulative expert probability mass at various temperatures.

Goal: understand how temperature affects the "width" (spread) of the
cumulative mass curves — i.e., whether higher T makes routing more diffuse.
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import json
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.special import softmax


def compute_cumulative_mass(probs_3d):
    """Given (docs, layers, experts) probabilities, return normalized cumulative mass."""
    sorted_desc = np.sort(probs_3d, axis=2)[:, :, ::-1]
    cumsum = np.cumsum(sorted_desc, axis=2)
    total = cumsum[:, :, -1:]
    total = np.where(total == 0, 1.0, total)
    return cumsum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", required=True, help="Path to embeddings_logits.npy")
    parser.add_argument("--info-file", required=True, help="Path to info.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.info_file) as f:
        info = json.load(f)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]

    # Load logits: (num_docs, num_layers * num_experts)
    logits = np.load(args.logits_file).astype(np.float32)
    num_docs = logits.shape[0]
    logits = logits.reshape(num_docs, num_layers, num_experts)
    print(f"Loaded logits: shape={logits.shape}")

    temperatures = sorted(args.temperatures)
    label = args.model_label or Path(args.logits_file).parent.name

    # Precompute cumulative mass for each temperature
    temp_cumsum = {}
    for T in temperatures:
        print(f"Computing T={T}...")
        probs = softmax(logits / T, axis=2)  # softmax over experts
        temp_cumsum[T] = compute_cumulative_mass(probs)

    # Color map for temperatures
    cmap = plt.cm.coolwarm
    temp_colors = {T: cmap(i / (len(temperatures) - 1)) for i, T in enumerate(temperatures)}

    # --- Plot 1: Full range, percentile bands per temperature ---
    fig, axes = plt.subplots(4, 4, figsize=(28, 24), dpi=100)
    fig.suptitle(
        f"Cumulative Expert Prob Mass at Various Temperatures\n{label}", fontsize=16, y=0.98
    )

    x = np.arange(1, num_experts + 1)

    for layer_idx in range(num_layers):
        ax = axes[layer_idx // 4, layer_idx % 4]

        for T in temperatures:
            y_data = temp_cumsum[T][:, layer_idx, :]  # (docs, experts)
            median = np.percentile(y_data, 50, axis=0)
            p10 = np.percentile(y_data, 10, axis=0)
            p90 = np.percentile(y_data, 90, axis=0)
            color = temp_colors[T]

            ax.plot(x, median, color=color, lw=1.5, label=f"T={T}" if layer_idx == 0 else None)
            ax.fill_between(x, p10, p90, color=color, alpha=0.15)

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Top-k experts")
        ax.set_ylabel("Cumulative prob mass")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, min(num_experts + 1, 130))
        ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / "cumulative_mass_temperature_sweep.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- Plot 2: Zoomed to top-32 ---
    fig2, axes2 = plt.subplots(4, 4, figsize=(28, 24), dpi=100)
    fig2.suptitle(
        f"Cumulative Expert Prob Mass — Zoom top-32, Temperature Sweep\n{label}",
        fontsize=16,
        y=0.98,
    )

    x_zoom = np.arange(1, 33)

    for layer_idx in range(num_layers):
        ax = axes2[layer_idx // 4, layer_idx % 4]

        for T in temperatures:
            y_data = temp_cumsum[T][:, layer_idx, :32]
            median = np.percentile(y_data, 50, axis=0)
            p10 = np.percentile(y_data, 10, axis=0)
            p90 = np.percentile(y_data, 90, axis=0)
            color = temp_colors[T]

            ax.plot(x_zoom, median, color=color, lw=1.5, label=f"T={T}" if layer_idx == 0 else None)
            ax.fill_between(x_zoom, p10, p90, color=color, alpha=0.15)

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Top-k experts")
        ax.set_ylabel("Cumulative prob mass")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, 33)
        ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)

    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path2 = output_dir / "cumulative_mass_temperature_sweep_zoom32.png"
    fig2.savefig(out_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")

    # --- Plot 3: Summary — median k to reach 50%/90% mass vs temperature ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    fig3.suptitle(f"Experts Needed to Reach X% Prob Mass vs Temperature\n{label}", fontsize=14)

    for thresh_idx, (thresh, ax) in enumerate(zip([0.5, 0.9], axes3)):
        # For each temperature and layer, find median k to reach threshold
        for layer_idx in range(num_layers):
            ks = []
            for T in temperatures:
                y_data = temp_cumsum[T][:, layer_idx, :]  # (docs, experts)
                # For each doc, find first k where cumsum >= threshold
                k_per_doc = (y_data >= thresh).argmax(axis=1) + 1  # 1-indexed
                ks.append(np.median(k_per_doc))

            alpha = 0.3 + 0.7 * (layer_idx / (num_layers - 1))
            ax.plot(
                temperatures,
                ks,
                marker="o",
                markersize=4,
                alpha=alpha,
                label=f"L{layer_idx}" if thresh_idx == 0 else None,
            )

        ax.set_xlabel("Temperature")
        ax.set_ylabel(f"Median k for {int(thresh*100)}% mass")
        ax.set_title(f"{int(thresh*100)}% probability mass")
        ax.set_xscale("log")
        ax.set_xticks(temperatures)
        ax.set_xticklabels([str(t) for t in temperatures])

    axes3[0].legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()

    out_path3 = output_dir / "experts_needed_vs_temperature.png"
    fig3.savefig(out_path3, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out_path3}")


if __name__ == "__main__":
    main()
