"""
Analyze per-layer routing pattern diversity in optB binary embeddings.

For each MoE layer, reports:
  - Number of unique routing patterns (expert combinations)
  - Shannon entropy (bits)
  - Coverage: how many patterns needed to cover 50%/90% of documents
  - Active experts per document (min/max/mean)

Usage:
    python -u -m src.scripts.analysis.layer_pattern_analysis \
        --emb-file claude_outputs/analysis/router_clustering_pretraining/embeddings_optB_binary.npy
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze per-layer routing pattern diversity")
    parser.add_argument("--emb-file", required=True, help="Path to optB binary .npy embedding file")
    parser.add_argument("--num-layers", type=int, default=16, help="Total number of MoE layers")
    parser.add_argument("--num-experts", type=int, default=127, help="Number of experts per layer")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Save results as JSON (default: <emb_dir>/layer_pattern_analysis.json)",
    )
    args = parser.parse_args()

    print(f"Loading {args.emb_file} ...")
    emb = np.load(args.emb_file)
    N, D = emb.shape
    num_layers = args.num_layers
    num_experts = args.num_experts

    assert (
        D == num_layers * num_experts
    ), f"Dim {D} != {num_layers} * {num_experts} = {num_layers * num_experts}"
    print(f"  {N} documents, {num_layers} layers, {num_experts} experts per layer")
    print()

    reshaped = emb.reshape(N, num_layers, num_experts)

    results = []
    print(
        f"{'Layer':>5} | {'Unique':>7} | {'Entropy':>8} | {'Active/doc':>10} | {'Top-10 cover':>12} | {'Top-50 cover':>12}"
    )
    print("-" * 75)

    for layer_idx in range(num_layers):
        layer_data = reshaped[:, layer_idx, :]

        # Active experts per doc
        active_per_doc = layer_data.sum(axis=1)

        # Unique patterns
        patterns, counts = np.unique(layer_data, axis=0, return_counts=True)
        n_unique = len(patterns)

        # Shannon entropy
        probs = counts / counts.sum()
        entropy = -(probs * np.log2(probs)).sum()

        # Coverage: sorted descending
        sorted_counts = np.sort(counts)[::-1]
        cumsum = np.cumsum(sorted_counts) / N
        top10_cover = cumsum[min(9, len(cumsum) - 1)]
        top50_cover = cumsum[min(49, len(cumsum) - 1)]

        # How many patterns for 50% and 90% coverage
        n_for_50 = int(np.searchsorted(cumsum, 0.5)) + 1
        n_for_90 = int(np.searchsorted(cumsum, 0.9)) + 1

        print(
            f"  {layer_idx:>3} | {n_unique:>7} | {entropy:>7.2f}b | "
            f"{active_per_doc.mean():>5.1f} [{active_per_doc.min()}-{active_per_doc.max()}] | "
            f"{top10_cover:>10.1%} | {top50_cover:>10.1%}"
        )

        results.append(
            {
                "layer": layer_idx,
                "unique_patterns": n_unique,
                "entropy_bits": round(float(entropy), 3),
                "active_per_doc_mean": round(float(active_per_doc.mean()), 2),
                "active_per_doc_min": int(active_per_doc.min()),
                "active_per_doc_max": int(active_per_doc.max()),
                "top10_coverage": round(float(top10_cover), 4),
                "top50_coverage": round(float(top50_cover), 4),
                "patterns_for_50pct": n_for_50,
                "patterns_for_90pct": n_for_90,
            }
        )

    # Also report combined stats
    print()
    all_unique = np.unique(emb, axis=0).shape[0]
    print(f"Combined (all {num_layers} layers): {all_unique} unique rows out of {N} documents")

    # Save JSON
    out_path = args.output_json or os.path.join(
        os.path.dirname(args.emb_file), "layer_pattern_analysis.json"
    )
    output = {
        "emb_file": args.emb_file,
        "num_docs": N,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "combined_unique_rows": all_unique,
        "per_layer": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
