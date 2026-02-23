"""
Derive sparse embedding variants from existing dense embeddings.

Reads embeddings_logits.npy and/or embeddings_probs.npy and creates sparsified
versions by keeping only the top-k experts per layer (rest zeroed).

Usage:
    python -m src.scripts.analysis.compute_embeddings \
        --data-dir claude_outputs/analysis/router_clustering_pretraining

    # Only compute sparse logits
    python -m src.scripts.analysis.compute_embeddings \
        --data-dir claude_outputs/analysis/router_clustering_pretraining \
        --embeddings logits_sparse
"""

import argparse
import json
import logging
import os

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K_SPARSE = 32


def sparsify_top_k(emb: np.ndarray, num_layers: int, num_experts: int, k: int) -> np.ndarray:
    """Zero out all but the top-k experts per layer (by value, i.e. highest activated)."""
    N = emb.shape[0]
    reshaped = emb.reshape(N, num_layers, num_experts).copy()
    bottom_idx = np.argsort(reshaped, axis=2)[:, :, :-k]
    np.put_along_axis(reshaped, bottom_idx, 0.0, axis=2)
    return reshaped.reshape(N, -1)


DERIVATIONS = {
    "logits_sparse": {
        "source": "embeddings_logits.npy",
        "output": "embeddings_logits_sparse.npy",
        "description": "Sparse avg logits: top-32 experts per layer, rest zeroed",
    },
    "probs_sparse": {
        "source": "embeddings_probs.npy",
        "output": "embeddings_probs_sparse.npy",
        "description": "Sparse avg probs: top-32 experts per layer, rest zeroed",
    },
}


def main():
    all_names = sorted(DERIVATIONS.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing dense embeddings and info.json")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Defaults to --data-dir.")
    parser.add_argument("--embeddings", default="all",
                        help=f"Comma-separated types to compute, or 'all'. "
                             f"Available: {', '.join(all_names)} (default: all)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir

    if args.embeddings == "all":
        requested = all_names
    else:
        requested = [s.strip() for s in args.embeddings.split(",")]
        for name in requested:
            if name not in DERIVATIONS:
                parser.error(f"Unknown type '{name}'. Available: {', '.join(all_names)}")

    with open(os.path.join(args.data_dir, "info.json")) as f:
        info = json.load(f)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]

    for name in requested:
        spec = DERIVATIONS[name]
        src_path = os.path.join(args.data_dir, spec["source"])
        out_path = os.path.join(output_dir, spec["output"])

        logger.info(f"\n--- {name}: {spec['description']} ---")
        logger.info(f"  Source: {src_path}")

        emb = np.load(src_path).astype(np.float32)
        logger.info(f"  Loaded: shape={emb.shape}")

        sparse = sparsify_top_k(emb, num_layers, num_experts, TOP_K_SPARSE).astype(np.float16)

        # Verify sparsity
        reshaped = sparse.reshape(-1, num_layers, num_experts)
        nnz_per_layer = (reshaped != 0).sum(axis=2)
        assert (nnz_per_layer == TOP_K_SPARSE).all(), \
            f"Expected {TOP_K_SPARSE} non-zeros per layer, got min={nnz_per_layer.min()}"

        density = (sparse != 0).mean()
        np.save(out_path, sparse)
        logger.info(f"  Saved: {out_path}")
        logger.info(f"  shape={sparse.shape}, dtype={sparse.dtype}, "
                    f"density={density:.1%} ({TOP_K_SPARSE}/{num_experts} per layer)")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
