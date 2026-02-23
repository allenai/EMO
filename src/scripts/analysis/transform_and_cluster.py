"""
Load a router embedding file and apply a named transformation pipeline.

This script is a boilerplate entrypoint for experimenting with different
dimensionality reduction, normalization, and clustering strategies on top
of extracted router embeddings.

Usage:
    python -m src.scripts.analysis.transform_and_cluster \
        --data-dir claude_outputs/analysis/router_clustering_pretraining \
        --embedding logits \
        --transform pca_l2

    # List available embeddings and transforms
    python -m src.scripts.analysis.transform_and_cluster --list
"""

import argparse
import json
import logging
import os

# Limit BLAS threads before numpy/sklearn imports
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

# Maps embedding name -> filename
EMBEDDING_FILES = {
    "logits":         "embeddings_logits.npy",
    "probs":          "embeddings_probs.npy",
    "logits_sparse":  "embeddings_logits_sparse.npy",
    "probs_sparse":   "embeddings_probs_sparse.npy",
}


def load_embedding(data_dir: str, embedding_name: str) -> tuple:
    """
    Load an embedding array and its associated metadata.

    Returns:
        emb: (N, D) float32 array
        meta: list of dicts (one per doc)
        info: dict with model/extraction metadata
    """
    if embedding_name not in EMBEDDING_FILES:
        raise ValueError(
            f"Unknown embedding '{embedding_name}'. "
            f"Available: {', '.join(sorted(EMBEDDING_FILES))}"
        )

    emb_path = os.path.join(data_dir, EMBEDDING_FILES[embedding_name])
    info_path = os.path.join(data_dir, "info.json")
    meta_path = os.path.join(data_dir, "metadata.jsonl.gz")

    logger.info(f"Loading embedding '{embedding_name}' from {emb_path}")
    emb = np.load(emb_path).astype(np.float32)

    with open(info_path) as f:
        info = json.load(f)

    import gzip
    meta = []
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    assert emb.shape[0] == len(meta), \
        f"Embedding rows ({emb.shape[0]}) != metadata rows ({len(meta)})"

    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    assert emb.shape[1] == num_layers * num_experts, \
        f"Embedding dim ({emb.shape[1]}) != {num_layers} * {num_experts}"

    logger.info(f"  shape={emb.shape}, dtype=float32, "
                f"range=[{emb.min():.4f}, {emb.max():.4f}]")
    logger.info(f"  {len(meta)} docs, {num_layers} layers, {num_experts} experts")

    return emb, meta, info


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

# Each transform is a function: (emb, info) -> transformed_emb
# `info` dict has num_layers, num_standard_experts, etc.
TRANSFORM_REGISTRY = {}


def register_transform(name: str, description: str):
    """Decorator to register a transform function."""
    def decorator(fn):
        TRANSFORM_REGISTRY[name] = {"fn": fn, "description": description}
        return fn
    return decorator


@register_transform("identity", "No transform — raw embedding values as-is")
def transform_identity(emb: np.ndarray, info: dict) -> np.ndarray:
    return emb


@register_transform("l2", "L2 normalize each document vector")
def transform_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.preprocessing import normalize
    return normalize(emb, norm="l2")


@register_transform("pca_l2", "PCA (95% variance) then L2 normalize")
def transform_pca_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    n_components = min(emb.shape[0], emb.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(emb)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")

    pca_k = PCA(n_components=k, random_state=42)
    reduced = pca_k.fit_transform(emb)
    return normalize(reduced, norm="l2")


def apply_transform(emb: np.ndarray, transform_name: str, info: dict) -> np.ndarray:
    """Apply a named transform to an embedding array."""
    if transform_name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{transform_name}'. "
            f"Available: {', '.join(sorted(TRANSFORM_REGISTRY))}"
        )
    logger.info(f"Applying transform: {transform_name}")
    return TRANSFORM_REGISTRY[transform_name]["fn"](emb, info)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load router embeddings and apply transformations."
    )
    parser.add_argument("--data-dir", type=str,
                        default="claude_outputs/analysis/router_clustering_pretraining",
                        help="Directory containing embeddings, metadata, and info.json")
    parser.add_argument("--embedding", type=str, default="logits",
                        help=f"Embedding type to load. "
                             f"Available: {', '.join(sorted(EMBEDDING_FILES))} "
                             f"(default: logits)")
    parser.add_argument("--transform", type=str, default="identity",
                        help=f"Transform to apply. "
                             f"Available: {', '.join(sorted(TRANSFORM_REGISTRY))} "
                             f"(default: identity)")
    parser.add_argument("--list", action="store_true",
                        help="List available embeddings and transforms, then exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable embeddings:")
        for name, filename in sorted(EMBEDDING_FILES.items()):
            print(f"  {name:20s} -> {filename}")
        print("\nAvailable transforms:")
        for name, entry in sorted(TRANSFORM_REGISTRY.items()):
            print(f"  {name:20s} — {entry['description']}")
        return

    # Load
    emb, meta, info = load_embedding(args.data_dir, args.embedding)

    # Transform
    transformed = apply_transform(emb, args.transform, info)
    logger.info(f"  Output shape: {transformed.shape}")

    # TODO: add clustering, saving, visualization, etc.
    logger.info("Done.")


if __name__ == "__main__":
    main()
