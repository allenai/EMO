"""
Derive embeddings from raw router logits and preprocess for clustering.

Two modes of operation:

1. **Derive**: Convert raw logits into different representations.
   Token-level: probs (softmax), topk_binary (top-k mask)
   Document-level: aggregate any token embedding via doc_boundaries.npy

2. **Preprocess**: Apply dimensionality reduction / normalization
   before clustering (mean-centering, PCA, L2 normalize, etc.)

Usage:
    # Derive token-level probs from logits
    python -m src.scripts.clustering.transform \\
        --data-dir .../pretraining/<model>/ --derive probs

    # Derive document-level topk_freq (aggregate topk_binary per doc)
    python -m src.scripts.clustering.transform \\
        --data-dir .../pretraining/<model>/ --derive doc_topk_freq

    # List available derivations and preprocessors
    python -m src.scripts.clustering.transform --list
"""

import argparse
import json
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derivation registry
# ---------------------------------------------------------------------------

DERIVE_REGISTRY = {}


def register_derive(name: str, description: str, output_file: str):
    """Decorator to register a derivation function."""

    def decorator(fn):
        DERIVE_REGISTRY[name] = {
            "fn": fn,
            "description": description,
            "output_file": output_file,
        }
        return fn

    return decorator


@register_derive("probs", "Per-token softmax probabilities", "embeddings_probs.npy")
def derive_probs(data_dir: str, info: dict) -> np.ndarray:
    """Apply softmax per layer to raw logits."""
    from scipy.special import softmax

    logits = np.load(os.path.join(data_dir, "embeddings_logits.npy")).astype(np.float32)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    N = logits.shape[0]

    reshaped = logits.reshape(N, num_layers, num_experts)
    probs = softmax(reshaped, axis=2)  # softmax over experts per layer
    result = probs.reshape(N, -1).astype(np.float16)

    logger.info(
        f"  Derived probs: shape={result.shape}, "
        f"per-layer sums ~{probs[0].sum(axis=1).mean():.4f}"
    )
    return result


@register_derive("topk_binary", "Binary top-k expert mask per token", "embeddings_topk_binary.npy")
def derive_topk_binary(data_dir: str, info: dict) -> np.ndarray:
    """Binary mask of top-k experts per layer per token."""
    logits = np.load(os.path.join(data_dir, "embeddings_logits.npy")).astype(np.float32)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    routed_top_k = info["routed_top_k"]
    N = logits.shape[0]

    reshaped = logits.reshape(N, num_layers, num_experts)
    # Get top-k indices per layer
    top_indices = np.argpartition(-reshaped, routed_top_k, axis=2)[:, :, :routed_top_k]
    binary = np.zeros_like(reshaped, dtype=np.uint8)
    np.put_along_axis(binary, top_indices, 1, axis=2)
    result = binary.reshape(N, -1)

    logger.info(
        f"  Derived topk_binary: shape={result.shape}, "
        f"top_k={routed_top_k}, density={result.mean():.4f}"
    )
    return result


@register_derive(
    "layer0_probs",
    "Per-token softmax probabilities for layer 0 only (num_experts dims)",
    "embeddings_layer0_probs.npy",
)
def derive_layer0_probs(data_dir: str, info: dict) -> np.ndarray:
    """Softmax over experts for just the first MoE layer's router logits."""
    from scipy.special import softmax

    logits = np.load(os.path.join(data_dir, "embeddings_logits.npy")).astype(np.float32)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    N = logits.shape[0]

    reshaped = logits.reshape(N, num_layers, num_experts)
    layer0 = reshaped[:, 0, :]  # (N, num_experts)
    probs = softmax(layer0, axis=1)
    result = probs.astype(np.float16)

    logger.info(
        f"  Derived layer0_probs: shape={result.shape}, "
        f"row sums ~{probs[0].sum():.4f}"
    )
    return result


@register_derive(
    "doc_probs", "Document-level mean softmax probabilities", "embeddings_doc_probs.npy"
)
def derive_doc_probs(data_dir: str, info: dict) -> np.ndarray:
    """Average token-level probs per document."""
    probs_path = os.path.join(data_dir, "embeddings_probs.npy")
    if not os.path.exists(probs_path):
        logger.info("  embeddings_probs.npy not found, deriving probs first...")
        probs = derive_probs(data_dir, info)
        np.save(probs_path, probs)
    return _aggregate_to_docs(data_dir, probs_path, np.float16)


@register_derive(
    "doc_layer0_probs",
    "Document-level mean layer-0 softmax (num_experts dims)",
    "embeddings_doc_layer0_probs.npy",
)
def derive_doc_layer0_probs(data_dir: str, info: dict) -> np.ndarray:
    """Average layer-0 token probs per document."""
    probs_path = os.path.join(data_dir, "embeddings_layer0_probs.npy")
    if not os.path.exists(probs_path):
        logger.info("  embeddings_layer0_probs.npy not found, deriving layer0_probs first...")
        probs = derive_layer0_probs(data_dir, info)
        np.save(probs_path, probs)
    return _aggregate_to_docs(data_dir, probs_path, np.float16)


@register_derive(
    "layer15_probs",
    "Per-token softmax probabilities for layer 15 only (num_experts dims)",
    "embeddings_layer15_probs.npy",
)
def derive_layer15_probs(data_dir: str, info: dict) -> np.ndarray:
    """Softmax over experts for just layer 15's router logits."""
    from scipy.special import softmax

    logits = np.load(os.path.join(data_dir, "embeddings_logits.npy")).astype(np.float32)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    N = logits.shape[0]

    reshaped = logits.reshape(N, num_layers, num_experts)
    layer15 = reshaped[:, 15, :]  # (N, num_experts)
    probs = softmax(layer15, axis=1)
    result = probs.astype(np.float16)

    logger.info(
        f"  Derived layer15_probs: shape={result.shape}, row sums ~{probs[0].sum():.4f}"
    )
    return result


@register_derive(
    "doc_layer15_probs",
    "Document-level mean layer-15 softmax (num_experts dims)",
    "embeddings_doc_layer15_probs.npy",
)
def derive_doc_layer15_probs(data_dir: str, info: dict) -> np.ndarray:
    """Average layer-15 token probs per document."""
    probs_path = os.path.join(data_dir, "embeddings_layer15_probs.npy")
    if not os.path.exists(probs_path):
        logger.info("  embeddings_layer15_probs.npy not found, deriving layer15_probs first...")
        probs = derive_layer15_probs(data_dir, info)
        np.save(probs_path, probs)
    return _aggregate_to_docs(data_dir, probs_path, np.float16)


@register_derive("doc_logits", "Document-level mean logits", "embeddings_doc_logits.npy")
def derive_doc_logits(data_dir: str, info: dict) -> np.ndarray:
    """Average token-level logits per document."""
    logits_path = os.path.join(data_dir, "embeddings_logits.npy")
    return _aggregate_to_docs(data_dir, logits_path, np.float16)


@register_derive(
    "doc_topk_freq", "Document-level top-k selection frequency", "embeddings_doc_topk_freq.npy"
)
def derive_doc_topk_freq(data_dir: str, info: dict) -> np.ndarray:
    """Average token-level topk_binary per document = selection frequency."""
    binary_path = os.path.join(data_dir, "embeddings_topk_binary.npy")
    if not os.path.exists(binary_path):
        logger.info("  embeddings_topk_binary.npy not found, deriving topk_binary first...")
        binary = derive_topk_binary(data_dir, info)
        np.save(binary_path, binary)
    return _aggregate_to_docs(data_dir, binary_path, np.float32)


def _aggregate_to_docs(data_dir: str, emb_path: str, out_dtype) -> np.ndarray:
    """Average token embeddings per document using doc_boundaries."""
    boundaries = np.load(os.path.join(data_dir, "doc_boundaries.npy"))
    token_emb = np.load(emb_path)
    num_docs = len(boundaries) - 1
    D = token_emb.shape[1]

    logger.info(f"  Aggregating {token_emb.shape[0]} tokens -> {num_docs} docs")
    doc_emb = np.zeros((num_docs, D), dtype=np.float32)
    for i in range(num_docs):
        start, end = boundaries[i], boundaries[i + 1]
        if end > start:
            doc_emb[i] = token_emb[start:end].astype(np.float32).mean(axis=0)

    result = doc_emb.astype(out_dtype)
    logger.info(f"  Result: shape={result.shape}, dtype={result.dtype}")
    return result


# ---------------------------------------------------------------------------
# Preprocessing registry (for clustering)
# ---------------------------------------------------------------------------

PREPROCESS_REGISTRY = {}


def register_preprocess(name: str, description: str):
    """Decorator to register a preprocessing transform."""

    def decorator(fn):
        PREPROCESS_REGISTRY[name] = {"fn": fn, "description": description}
        return fn

    return decorator


@register_preprocess("identity", "No preprocessing — raw values as-is")
def preprocess_identity(emb: np.ndarray, info: dict) -> np.ndarray:
    return emb


@register_preprocess("l2", "L2 normalize each vector")
def preprocess_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.preprocessing import normalize

    return normalize(emb, norm="l2")


_VARIANCE_CUTOFF_SAMPLE_SIZE = 1_000_000


def _find_variance_cutoff_k(centered: np.ndarray, variance: float = 0.95) -> int:
    """Find #components explaining `variance` fraction using a 1M-row subsample.

    Full-rank PCA on 20M+ rows OOMs; the top-component variance ratios are
    stable under subsampling, so we fit on a sample and reuse the cutoff.
    """
    from sklearn.decomposition import PCA

    N = centered.shape[0]
    if N > _VARIANCE_CUTOFF_SAMPLE_SIZE:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, _VARIANCE_CUTOFF_SAMPLE_SIZE, replace=False)
        sample = centered[idx]
        logger.info(
            f"  Variance-cutoff PCA on {_VARIANCE_CUTOFF_SAMPLE_SIZE:,}-row subsample "
            f"(full data has {N:,} rows)"
        )
    else:
        sample = centered

    n_components = min(sample.shape[0], sample.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(sample)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance)) + 1
    logger.info(f"  PCA: {k} components explain {cumvar[k-1]:.1%} variance")
    return k


@register_preprocess("mean_pca", "Mean-center then PCA (95% variance)")
def preprocess_mean_pca(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA

    centered = emb - emb.mean(axis=0, keepdims=True)
    k = _find_variance_cutoff_k(centered)
    pca_k = PCA(n_components=k, svd_solver="randomized", random_state=42)
    return pca_k.fit_transform(centered)


@register_preprocess("mean_pca_l2", "Mean-center, PCA (95% variance), L2 normalize")
def preprocess_mean_pca_l2(emb: np.ndarray, info: dict) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    centered = emb - emb.mean(axis=0, keepdims=True)
    k = _find_variance_cutoff_k(centered)
    pca_k = PCA(n_components=k, svd_solver="randomized", random_state=42)
    reduced = pca_k.fit_transform(centered)
    return normalize(reduced, norm="l2")


def apply_preprocess(emb: np.ndarray, name: str, info: dict) -> np.ndarray:
    """Apply a named preprocessing transform."""
    if name not in PREPROCESS_REGISTRY:
        raise ValueError(
            f"Unknown preprocess '{name}'. " f"Available: {', '.join(sorted(PREPROCESS_REGISTRY))}"
        )
    logger.info(f"Preprocessing: {name}")
    return PREPROCESS_REGISTRY[name]["fn"](emb, info)


# ---------------------------------------------------------------------------
# Embedding loading (knows about all derived types)
# ---------------------------------------------------------------------------

# Maps name -> filename for all possible embeddings (raw + derived)
EMBEDDING_FILES = {
    # Raw (from extract.py)
    "logits": "embeddings_logits.npy",
    # Token-level derived
    "probs": "embeddings_probs.npy",
    "topk_binary": "embeddings_topk_binary.npy",
    "layer0_probs": "embeddings_layer0_probs.npy",
    "layer15_probs": "embeddings_layer15_probs.npy",
    # Document-level derived
    "doc_logits": "embeddings_doc_logits.npy",
    "doc_probs": "embeddings_doc_probs.npy",
    "doc_topk_freq": "embeddings_doc_topk_freq.npy",
    "doc_layer0_probs": "embeddings_doc_layer0_probs.npy",
    "doc_layer15_probs": "embeddings_doc_layer15_probs.npy",
}


def load_embedding(data_dir: str, name: str) -> tuple:
    """
    Load an embedding and its metadata.

    Returns (emb, meta, info) where:
      emb: (N, D) float32 array
      meta: list of dicts (per-token or per-doc)
      info: dict with extraction config
    """
    if name not in EMBEDDING_FILES:
        raise ValueError(
            f"Unknown embedding '{name}'. " f"Available: {', '.join(sorted(EMBEDDING_FILES))}"
        )

    emb_path = os.path.join(data_dir, EMBEDDING_FILES[name])
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embedding file not found: {emb_path}\n"
            f"Run: python -m src.scripts.clustering.transform "
            f"--data-dir {data_dir} --derive {name}"
        )

    logger.info(f"Loading embedding '{name}' from {emb_path}")
    emb = np.load(emb_path).astype(np.float32)

    info_path = os.path.join(data_dir, "info.json")
    with open(info_path) as f:
        info = json.load(f)

    # Load appropriate metadata
    is_doc_level = name.startswith("doc_")
    if is_doc_level:
        meta_path = os.path.join(data_dir, "metadata_docs.jsonl.gz")
    else:
        meta_path = os.path.join(data_dir, "metadata_tokens.jsonl.gz")

    import gzip

    meta = []
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    assert emb.shape[0] == len(
        meta
    ), f"Embedding rows ({emb.shape[0]}) != metadata rows ({len(meta)})"

    logger.info(f"  shape={emb.shape}, range=[{emb.min():.4f}, {emb.max():.4f}]")
    return emb, meta, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Derive embeddings from logits and preprocess for clustering"
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory containing embeddings_logits.npy and info.json"
    )
    parser.add_argument(
        "--derive",
        type=str,
        default=None,
        choices=sorted(DERIVE_REGISTRY.keys()),
        help="Derivation to compute from raw logits",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available derivations and preprocessors"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable derivations (--derive):")
        for name, entry in sorted(DERIVE_REGISTRY.items()):
            print(f"  {name:20s} -> {entry['output_file']:40s}  {entry['description']}")
        print("\nAvailable preprocessors (used by cluster.py --preprocess):")
        for name, entry in sorted(PREPROCESS_REGISTRY.items()):
            print(f"  {name:20s}  {entry['description']}")
        print("\nAll loadable embeddings:")
        for name, filename in sorted(EMBEDDING_FILES.items()):
            print(f"  {name:20s} -> {filename}")
        return

    if not args.data_dir:
        parser.error("--data-dir is required")

    if args.derive:
        info_path = os.path.join(args.data_dir, "info.json")
        with open(info_path) as f:
            info = json.load(f)

        entry = DERIVE_REGISTRY[args.derive]
        logger.info(f"Deriving: {args.derive} ({entry['description']})")
        result = entry["fn"](args.data_dir, info)

        out_path = os.path.join(args.data_dir, entry["output_file"])
        np.save(out_path, result)
        logger.info(f"Saved: {out_path}  shape={result.shape}  dtype={result.dtype}")
    else:
        parser.error("Specify --derive or --list")


if __name__ == "__main__":
    main()
