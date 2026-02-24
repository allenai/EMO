import argparse
import json
import os

import torch

_parser = argparse.ArgumentParser(description="Analyze router probability files from MoE forward passes.")
_parser.add_argument(
    "--router-files", type=str, nargs="+", required=True, help="Paths to one or more *-router.jsonl files"
)
_parser.add_argument("--top-k", type=int, default=5, help="Number of top experts to report per layer")
_parser.add_argument("--visualize", action="store_true", help="Enable visualization (heatmap + optional bar charts)")
_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save plots")
_parser.add_argument(
    "--per-layer-bar", action="store_true", help="Also generate per-layer bar charts (requires --visualize)"
)


def load_jsonl_file(file_path):
    """Loads a jsonl file and returns a list of json objects."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_router_probs(filepath):
    """Load a router JSONL file and return a tensor of shape (num_layers, num_experts)."""
    data = load_jsonl_file(filepath)
    return torch.tensor(data[0]["avg_router_probabilities"])


def print_analysis(filepath, probs, top_k):
    """Print a formatted table of top-K experts per layer and global top-K."""
    basename = os.path.basename(filepath)
    num_layers = probs.shape[0]

    print(f"\n=== {basename} ===")
    print(f"Shape: {num_layers} layers x {probs.shape[1]} experts")
    print(f"{'Layer':<7}| Top-{top_k} Experts (idx: prob)")
    print(f"{'':->7}|{'':->60}")

    for layer_idx in range(num_layers):
        values, indices = torch.topk(probs[layer_idx], top_k)
        entries = ", ".join(f"{idx.item()}: {val.item():.4f}" for idx, val in zip(indices, values))
        print(f"{layer_idx:<7}| {entries}")

    # Global top-K (summed across layers)
    global_probs = probs.sum(dim=0)
    values, indices = torch.topk(global_probs, top_k)
    print(f"\nGlobal Top-{top_k} Experts (summed across layers):")
    for idx, val in zip(indices, values):
        print(f"  Expert {idx.item():>3}: {val.item():.4f}")


def plot_heatmaps(file_probs, top_k, output_dir):
    """Generate heatmap(s) of router probabilities, one subplot per file."""
    import matplotlib.pyplot as plt

    n_files = len(file_probs)
    fig, axes = plt.subplots(1, n_files, figsize=(8 * n_files, max(6, file_probs[0][1].shape[0] * 0.4)))
    if n_files == 1:
        axes = [axes]

    # Shared color range across all files
    vmin = min(p.min().item() for _, p in file_probs)
    vmax = max(p.max().item() for _, p in file_probs)

    for ax, (filepath, probs) in zip(axes, file_probs):
        im = ax.imshow(probs.numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(os.path.basename(filepath), fontsize=10)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")

        # Annotate top-K experts per layer with expert number labels
        for layer_idx in range(probs.shape[0]):
            _, indices = torch.topk(probs[layer_idx], top_k)
            for idx in indices:
                ax.text(
                    idx.item(), layer_idx, str(idx.item()),
                    ha="center", va="center", fontsize=4, color="red", fontweight="bold",
                )

    fig.colorbar(im, ax=axes, label="Router Probability", shrink=0.8)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "router_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved heatmap to {out_path}")


def plot_per_layer_bars(filepath, probs, top_k, output_dir):
    """Generate per-layer bar charts for a single file."""
    import matplotlib.pyplot as plt

    num_layers = probs.shape[0]
    num_experts = probs.shape[1]

    # Limit to a reasonable number of subplots per figure
    cols = 4
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_probs = probs[layer_idx].numpy()
        _, topk_indices = torch.topk(probs[layer_idx], top_k)
        topk_set = set(topk_indices.tolist())

        colors = ["#d62728" if i in topk_set else "#1f77b4" for i in range(num_experts)]
        ax.bar(range(num_experts), layer_probs, color=colors, width=1.0)
        # Label top-K bars with expert number
        for idx in topk_indices.tolist():
            ax.text(
                idx, layer_probs[idx], str(idx),
                ha="center", va="bottom", fontsize=5, fontweight="bold",
            )
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Expert", fontsize=7)
        ax.set_ylabel("Prob", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    basename = os.path.basename(filepath)
    fig.suptitle(f"Per-Layer Expert Probabilities: {basename}", fontsize=12)
    fig.tight_layout()
    stem = os.path.splitext(basename)[0]
    out_path = os.path.join(output_dir, f"router_bars_{stem}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to {out_path}")


def main():
    args = _parser.parse_args()

    # Load all files
    file_probs = []
    for filepath in args.router_files:
        probs = load_router_probs(filepath)
        file_probs.append((filepath, probs))

    # Print analysis for each file
    for filepath, probs in file_probs:
        print_analysis(filepath, probs, args.top_k)

    # Visualization
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_heatmaps(file_probs, args.top_k, args.output_dir)
        if args.per_layer_bar:
            for filepath, probs in file_probs:
                plot_per_layer_bars(filepath, probs, args.top_k, args.output_dir)


if __name__ == "__main__":
    main()
