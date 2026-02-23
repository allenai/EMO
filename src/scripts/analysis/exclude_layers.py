"""
Extract a subset of layers from router embedding .npy files.

Keeps only the specified layers (removing the rest), producing a lower-dimensional
embedding. Output files are named by the layers *kept* (e.g., _L1-15.npy, _L6-10.npy).

Usage:
    # Keep layers 1-15 (exclude layer 0)
    python -u -m src.scripts.analysis.exclude_layers \
        --emb-file embeddings_optB_binary.npy --keep-layers 1-15

    # Keep layers 6-10
    python -u -m src.scripts.analysis.exclude_layers \
        --emb-file embeddings_optB_binary.npy --keep-layers 6-10

    # Keep only layer 15
    python -u -m src.scripts.analysis.exclude_layers \
        --emb-file embeddings_optB_binary.npy --keep-layers 15
"""

import argparse
import os

import numpy as np


def parse_layer_spec(spec: str) -> list[int]:
    """Parse a layer spec like '15', '6-10', '1-15' into a sorted list of indices."""
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def make_suffix(kept: list[int]) -> str:
    """Generate a compact suffix like L15, L6-10, L1-15."""
    if len(kept) == 1:
        return f"L{kept[0]}"
    if kept == list(range(kept[0], kept[-1] + 1)):
        return f"L{kept[0]}-{kept[-1]}"
    # Non-contiguous
    return "L" + "_".join(str(i) for i in kept)


def main():
    parser = argparse.ArgumentParser(description="Extract layer subsets from router embeddings")
    parser.add_argument("--emb-file", required=True, help="Path to .npy embedding file")
    parser.add_argument("--keep-layers", required=True,
                        help="Layer indices to keep (e.g. '15', '6-10', '1-15', '0,5,10')")
    parser.add_argument("--num-layers", type=int, default=16, help="Total number of MoE layers")
    parser.add_argument("--num-experts", type=int, default=127, help="Number of experts per layer")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same directory as emb-file)")
    args = parser.parse_args()

    kept = parse_layer_spec(args.keep_layers)
    num_layers = args.num_layers
    num_experts = args.num_experts

    for idx in kept:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers})")

    print(f"Loading {args.emb_file} ...")
    emb = np.load(args.emb_file)
    orig_dtype = emb.dtype
    N, D = emb.shape
    expected_dim = num_layers * num_experts
    if D != expected_dim:
        raise ValueError(f"Embedding dim {D} != num_layers({num_layers}) * num_experts({num_experts}) = {expected_dim}")

    print(f"  Original shape: ({N}, {D}), dtype: {orig_dtype}")
    print(f"  Keeping layers: {kept}")

    reshaped = emb.reshape(N, num_layers, num_experts)
    selected = reshaped[:, kept, :]
    result = selected.reshape(N, len(kept) * num_experts)

    print(f"  Result shape: {result.shape} ({len(kept)} layers x {num_experts} experts)")
    assert result.dtype == orig_dtype, f"dtype changed: {orig_dtype} -> {result.dtype}"

    suffix = make_suffix(kept)
    base, ext = os.path.splitext(os.path.basename(args.emb_file))
    out_name = f"{base}_{suffix}{ext}"
    out_dir = args.output_dir or os.path.dirname(args.emb_file)
    out_path = os.path.join(out_dir, out_name)

    np.save(out_path, result)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
