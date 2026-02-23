"""
Extract per-document router embeddings from a trained MoE model.

For each document, computes embeddings of shape (num_layers * num_standard_experts,)
representing the router activations across all layers, concatenated.

Embedding types are registered via EMBEDDING_REGISTRY. Each type defines a function
that receives per-layer router logits (GPU tensors) and an attention mask, and
returns a (B, L*E) numpy array. This allows each type to operate on raw per-token
logits before any aggregation (e.g. softmax per token, then average).

Currently supported:
    logits  — average pre-softmax router logits per expert per layer (float16)
    probs   — per-token softmax, then average probabilities per expert per layer (float16)

New embedding types can be added by:
    1. Define a function with signature:
       (per_layer_logits: List[Tensor(B,S,E)], attention_mask: Tensor(B,S),
        num_layers: int, num_experts: int) -> np.ndarray(B, L*E)
    2. Register it in EMBEDDING_REGISTRY.

Proportional sampling: reads mix_composition.json (produced by analyze_data_mix.py)
to allocate tokens per source proportionally. Uses range-GET from S3 to avoid
downloading full files (which are multi-GB each).

Usage:
    # Extract all embedding types
    python -m src.scripts.analysis.extract_router_embeddings \\
        --model-path models/.../step30995-hf \\
        --composition-file claude_outputs/analysis/router_clustering/mix_composition.json \\
        --output-dir claude_outputs/analysis/router_clustering \\
        --target-tokens 20_000_000 \\
        --batch-size 32

    # Extract only logits
    python -m src.scripts.analysis.extract_router_embeddings \\
        --model-path ... --composition-file ... --output-dir ... \\
        --embeddings logits
"""

import argparse
import gzip
import json
import logging
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 100257
BYTES_PER_TOKEN = 4  # uint32


# ---------------------------------------------------------------------------
# Embedding type registry
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingType:
    """Definition of an embedding type."""
    name: str
    filename: str
    dtype: type
    compute_fn: Callable[[List[torch.Tensor], torch.Tensor, int, int], np.ndarray]
    description: str


def compute_logits_embedding(
    per_layer_logits: List[torch.Tensor],
    attention_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> np.ndarray:
    """
    Average pre-softmax router logits per expert per layer.

    For each document, averages the raw logits across all non-padding tokens.
    Shape: (B, num_layers * num_experts), dtype float16.
    """
    mask = attention_mask.unsqueeze(-1)  # (B, S, 1)
    doc_lens = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).float()  # (B, 1, 1)

    layers = []
    for logits in per_layer_logits:  # (B, S, E)
        avg = (logits.float() * mask).sum(dim=1) / doc_lens.squeeze(1)  # (B, E)
        layers.append(avg.cpu().numpy())

    return np.concatenate(layers, axis=1).astype(np.float16)


def compute_probs_embedding(
    per_layer_logits: List[torch.Tensor],
    attention_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> np.ndarray:
    """
    Per-token softmax probabilities, averaged per expert per layer.

    Applies softmax over experts for each token independently, then averages
    across non-padding tokens. Each layer block sums to ~1.
    Shape: (B, num_layers * num_experts), dtype float16.
    """
    mask = attention_mask.unsqueeze(-1)  # (B, S, 1)
    doc_lens = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).half()  # (B, 1, 1)

    layers = []
    for logits in per_layer_logits:  # (B, S, E)
        probs = F.softmax(logits.float(), dim=-1).half()  # (B, S, E)
        avg = (probs * mask).sum(dim=1) / doc_lens.squeeze(1)  # (B, E)
        layers.append(avg.cpu().numpy())

    return np.concatenate(layers, axis=1).astype(np.float16)


TOP_K_SPARSE = 32  # number of experts to keep per layer in sparse variants


def _sparsify_top_k(emb: np.ndarray, num_layers: int, num_experts: int, k: int) -> np.ndarray:
    """Zero out all but the top-k experts per layer (by value, i.e. highest activated)."""
    B = emb.shape[0]
    reshaped = emb.reshape(B, num_layers, num_experts).copy()
    bottom_idx = np.argsort(reshaped, axis=2)[:, :, :-k]
    np.put_along_axis(reshaped, bottom_idx, 0.0, axis=2)
    return reshaped.reshape(B, -1)


def compute_logits_sparse_embedding(
    per_layer_logits: List[torch.Tensor],
    attention_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> np.ndarray:
    """
    Sparse average logits: top-32 experts per layer, rest zeroed.

    Computes average logits first, then keeps only the top-32 experts per layer
    by absolute value.
    Shape: (B, num_layers * num_experts), dtype float16, ~25% non-zero.
    """
    dense = compute_logits_embedding(per_layer_logits, attention_mask, num_layers, num_experts)
    return _sparsify_top_k(dense.astype(np.float32), num_layers, num_experts, TOP_K_SPARSE).astype(np.float16)


def compute_probs_sparse_embedding(
    per_layer_logits: List[torch.Tensor],
    attention_mask: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> np.ndarray:
    """
    Sparse average probabilities: top-32 experts per layer, rest zeroed.

    Computes per-token softmax then average first, then keeps only the top-32
    experts per layer by value.
    Shape: (B, num_layers * num_experts), dtype float16, ~25% non-zero.
    """
    dense = compute_probs_embedding(per_layer_logits, attention_mask, num_layers, num_experts)
    return _sparsify_top_k(dense.astype(np.float32), num_layers, num_experts, TOP_K_SPARSE).astype(np.float16)


EMBEDDING_REGISTRY: Dict[str, EmbeddingType] = {
    "logits": EmbeddingType(
        name="logits",
        filename="embeddings_logits.npy",
        dtype=np.float16,
        compute_fn=compute_logits_embedding,
        description="Average pre-softmax router logits per expert per layer",
    ),
    "probs": EmbeddingType(
        name="probs",
        filename="embeddings_probs.npy",
        dtype=np.float16,
        compute_fn=compute_probs_embedding,
        description="Per-token softmax probabilities, averaged per expert per layer",
    ),
    "logits_sparse": EmbeddingType(
        name="logits_sparse",
        filename="embeddings_logits_sparse.npy",
        dtype=np.float16,
        compute_fn=compute_logits_sparse_embedding,
        description="Sparse avg logits: top-32 experts per layer, rest zeroed",
    ),
    "probs_sparse": EmbeddingType(
        name="probs_sparse",
        filename="embeddings_probs_sparse.npy",
        dtype=np.float16,
        compute_fn=compute_probs_sparse_embedding,
        description="Sparse avg probs: top-32 experts per layer, rest zeroed",
    ),
}


# ---------------------------------------------------------------------------
# S3 data loading
# ---------------------------------------------------------------------------

def stream_bytes_from_s3(s3_path: str, num_bytes: int) -> bytes:
    """Range-GET the first `num_bytes` of an S3 file into memory."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "aws", "s3api", "get-object",
                "--bucket", "ai2-llm",
                "--key", s3_path.replace("s3://ai2-llm/", ""),
                "--range", f"bytes=0-{num_bytes - 1}",
                tmp_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"S3 range-GET failed for {s3_path}: {result.stderr[:200]}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def tokens_from_bytes(raw: bytes) -> np.ndarray:
    """Parse headerless raw uint32 binary into token IDs."""
    n = len(raw) // BYTES_PER_TOKEN
    return np.frombuffer(raw[: n * BYTES_PER_TOKEN], dtype=np.uint32).astype(np.int32)


def iter_documents(
    tokens: np.ndarray, min_len: int = 32, max_len: int = 2048
) -> Iterator[np.ndarray]:
    """Split a token array on EOS tokens, yielding individual documents."""
    eos_pos = np.where(tokens == EOS_TOKEN_ID)[0]
    start = 0
    for pos in eos_pos:
        doc = tokens[start : pos + 1]
        if min_len <= len(doc) <= max_len:
            yield doc
        start = pos + 1


def load_source_documents(
    s3_files: List[str],
    target_tokens: int,
    min_doc_len: int,
    max_doc_len: int,
) -> List[np.ndarray]:
    """
    Stream just enough data from S3 (across files if needed) to collect
    `target_tokens` worth of documents for a single source.
    """
    docs = []
    collected_tokens = 0
    bytes_needed = int(target_tokens * BYTES_PER_TOKEN * 1.5)

    for s3_path in s3_files:
        if collected_tokens >= target_tokens:
            break
        remaining_bytes = max(0, bytes_needed - collected_tokens * BYTES_PER_TOKEN)
        logger.info(f"  Streaming {remaining_bytes / 1e6:.1f} MB from {s3_path.split('/')[-1]} ...")
        try:
            raw = stream_bytes_from_s3(s3_path, remaining_bytes)
        except Exception as e:
            logger.warning(f"  Skipping {s3_path}: {e}")
            continue

        tokens = tokens_from_bytes(raw)
        for doc in iter_documents(tokens, min_doc_len, max_doc_len):
            docs.append(doc.copy())
            collected_tokens += len(doc)
            if collected_tokens >= target_tokens:
                break

    logger.info(f"  Collected {len(docs)} docs / {collected_tokens:,} tokens")
    return docs


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def embed_batch(
    model,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
    embedding_types: List[EmbeddingType],
) -> Dict[str, np.ndarray]:
    """
    Run a batch of documents through the model and compute requested embeddings.

    Phase 1 (GPU): Single forward pass, collect per-layer logits as tensors.
    Phase 2: Each embedding type's compute_fn transforms the logits.

    Returns dict mapping embedding name -> (B, num_layers * num_standard_experts) array.
    """
    B = len(batch_docs)
    max_len = max(len(d) for d in batch_docs)
    input_ids = torch.full((B, max_len), EOS_TOKEN_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, doc in enumerate(batch_docs):
        L = len(doc)
        input_ids[i, :L] = torch.from_numpy(doc.astype(np.int64)).to(device)
        attention_mask[i, :L] = 1

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_router_logits=True,
    )

    # Collect per-layer logits: List of (B, S, num_standard_experts)
    per_layer_logits = []
    for layer_logits in outputs.router_logits:
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts]
        per_layer_logits.append(logits)

    # Compute each requested embedding type
    results = {}
    for et in embedding_types:
        results[et.name] = et.compute_fn(
            per_layer_logits, attention_mask, num_layers, num_standard_experts
        )

    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    model, tokenizer, source_doc_samples, device,
    num_layers, num_standard_experts, embedding_types,
):
    logger.info("=== Running sanity checks ===")

    sample_doc = source_doc_samples[0][1][0]
    D = num_layers * num_standard_experts

    # Check 1: shapes and dtypes
    results = embed_batch(
        model, [sample_doc], device, num_layers, num_standard_experts, embedding_types
    )
    for et in embedding_types:
        arr = results[et.name]
        assert arr.shape == (1, D), f"{et.name} shape {arr.shape} != (1, {D})"
        assert arr.dtype == et.dtype, f"{et.name} dtype {arr.dtype} != {et.dtype}"
        assert not np.isnan(arr).any(), f"{et.name} contains NaN"
        logger.info(f"  CHECK 1 [{et.name}]: shape=(1, {D}), dtype={et.dtype}, "
                    f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    logger.info(f"  CHECK 1 PASSED")

    # Check 2: probs sum to ~1 per layer
    if "probs" in results:
        per_layer_sum = results["probs"][0].reshape(num_layers, num_standard_experts).sum(axis=1)
        assert (per_layer_sum > 0.95).all() and (per_layer_sum < 1.05).all(), \
            f"Softmax row sums off: {per_layer_sum}"
        logger.info(f"  CHECK 2 PASSED: probs per-layer sums in [0.95, 1.05]")

    # Check 3: decoded text looks sane
    logger.info("  CHECK 3: decoding sample docs per source")
    for label, docs in source_doc_samples[:3]:
        decoded = tokenizer.decode(docs[0][:80].tolist(), skip_special_tokens=True)
        logger.info(f"    [{label}] {decoded[:120]!r}")

    # Check 4: same-source docs more similar than cross-source
    if len(source_doc_samples) >= 2:
        def cos(x, y):
            x, y = x.astype(np.float32), y.astype(np.float32)
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

        check_et = embedding_types[0]
        la, da = source_doc_samples[0]
        lb, db = source_doc_samples[1]
        if len(da) >= 2 and len(db) >= 1:
            r0 = embed_batch(model, [da[0]], device, num_layers, num_standard_experts, [check_et])
            r1 = embed_batch(model, [da[1]], device, num_layers, num_standard_experts, [check_et])
            r2 = embed_batch(model, [db[0]], device, num_layers, num_standard_experts, [check_et])
            same = cos(r0[check_et.name][0], r1[check_et.name][0])
            cross = cos(r0[check_et.name][0], r2[check_et.name][0])
            logger.info(f"  CHECK 4 ({check_et.name}): same-source ({la}) cos={same:.3f}  "
                        f"cross-source ({la} vs {lb}) cos={cross:.3f}")
            if same > cross:
                logger.info("  CHECK 4 PASSED: same-source more similar")
            else:
                logger.info("  CHECK 4 NOTE: cross-source not lower — may be fine")

    logger.info("=== Sanity checks complete ===\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_names = sorted(EMBEDDING_REGISTRY.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--composition-file", required=True,
                        help="mix_composition.json from analyze_data_mix.py")
    parser.add_argument("--output-dir", default="claude_outputs/analysis/router_clustering")
    parser.add_argument("--target-tokens", type=int, default=20_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-doc-len", type=int, default=2048)
    parser.add_argument("--min-doc-len", type=int, default=32)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--embeddings", default="all",
                        help=f"Comma-separated embedding types to compute, or 'all'. "
                             f"Available: {', '.join(all_names)} (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve which embedding types to compute
    if args.embeddings == "all":
        requested_names = all_names
    else:
        requested_names = [s.strip() for s in args.embeddings.split(",")]
        for name in requested_names:
            if name not in EMBEDDING_REGISTRY:
                parser.error(f"Unknown embedding type '{name}'. "
                             f"Available: {', '.join(all_names)}")

    embedding_types = [EMBEDDING_REGISTRY[name] for name in requested_names]
    logger.info(f"Will compute embeddings: {[et.name for et in embedding_types]}")

    # Load proportional composition
    with open(args.composition_file) as f:
        composition = json.load(f)

    sources = composition["sources"]
    logger.info(f"Loaded composition for {len(sources)} sources")
    for label, info in sorted(sources.items(), key=lambda x: -x[1]["fraction"]):
        alloc = int(info["fraction"] * args.target_tokens)
        logger.info(f"  {label}: {info['fraction']:.2%} -> {alloc:,} tokens")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = str(next(model.parameters()).device)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_all_experts = cfg.num_experts if hasattr(cfg, "num_experts") else cfg.num_local_experts
    num_shared = getattr(cfg, "num_shared_experts", 0)
    num_standard_experts = num_all_experts - num_shared
    emb_dim = num_layers * num_standard_experts

    logger.info(f"Model: {num_layers} layers, {num_standard_experts} standard experts")
    logger.info(f"Embedding dim: {emb_dim}")

    # Collect a few docs per source for sanity checks
    source_doc_samples = []
    for label, info in sources.items():
        raw = stream_bytes_from_s3(info["all_files"][0], 3_000_000)
        toks = tokens_from_bytes(raw)
        docs = list(iter_documents(toks, args.min_doc_len, args.max_doc_len))[:5]
        if docs:
            source_doc_samples.append((label, docs))

    run_sanity_checks(
        model, tokenizer, source_doc_samples, device,
        num_layers, num_standard_experts, embedding_types,
    )

    if args.sanity_check_only:
        logger.info("--sanity-check-only specified, exiting.")
        return

    # Main extraction: proportional sampling per source
    all_embeddings: Dict[str, List[np.ndarray]] = {et.name: [] for et in embedding_types}
    all_meta: List[dict] = []

    for label, info in sorted(sources.items(), key=lambda x: -x[1]["fraction"]):
        target_source_tokens = int(info["fraction"] * args.target_tokens)
        if target_source_tokens < args.min_doc_len:
            logger.info(f"Skipping {label}: allocation too small ({target_source_tokens} tokens)")
            continue

        logger.info(f"\nSource: {label}  target={target_source_tokens:,} tokens")
        docs = load_source_documents(
            info["all_files"], target_source_tokens, args.min_doc_len, args.max_doc_len
        )

        if not docs:
            logger.warning(f"  No documents collected for {label}, skipping")
            continue

        # Batch inference
        batch_embeddings: Dict[str, List[np.ndarray]] = {et.name: [] for et in embedding_types}
        num_batches = (len(docs) + args.batch_size - 1) // args.batch_size
        t0 = time.time()
        for batch_idx, i in enumerate(range(0, len(docs), args.batch_size)):
            batch = docs[i : i + args.batch_size]
            results = embed_batch(
                model, batch, device, num_layers, num_standard_experts, embedding_types
            )
            for name, arr in results.items():
                batch_embeddings[name].append(arr)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{label}] batch {batch_idx+1}/{num_batches} "
                    f"({(batch_idx+1)/num_batches:.0%})  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
                )

        for name in batch_embeddings:
            all_embeddings[name].append(np.concatenate(batch_embeddings[name], axis=0))

        for doc in docs:
            try:
                preview = tokenizer.decode(doc[:800].tolist(), skip_special_tokens=True)[:3000]
            except Exception:
                preview = ""
            all_meta.append({"source": label, "doc_len": int(len(doc)), "preview": preview})

        logger.info(f"  Done: {len(docs)} docs embedded for {label}")

    # Save embeddings
    for et in embedding_types:
        out_path = os.path.join(args.output_dir, et.filename)
        arr = np.concatenate(all_embeddings[et.name], axis=0)
        np.save(out_path, arr)
        logger.info(f"  {et.name}: {out_path}  shape={arr.shape}  dtype={arr.dtype}")

    # Save metadata
    out_meta = os.path.join(args.output_dir, "metadata.jsonl.gz")
    with gzip.open(out_meta, "wt") as f:
        for m in all_meta:
            f.write(json.dumps(m) + "\n")

    # Save info
    source_counts = defaultdict(int)
    for m in all_meta:
        source_counts[m["source"]] += 1

    info_out = {
        "num_docs": len(all_meta),
        "total_tokens": int(sum(m["doc_len"] for m in all_meta)),
        "num_layers": num_layers,
        "num_standard_experts": num_standard_experts,
        "emb_dim": emb_dim,
        "model_path": args.model_path,
        "embedding_types": [et.name for et in embedding_types],
        "source_doc_counts": dict(source_counts),
    }
    out_info = os.path.join(args.output_dir, "info.json")
    with open(out_info, "w") as f:
        json.dump(info_out, f, indent=2)

    logger.info(f"\nSaved {len(all_meta)} document embeddings")
    logger.info(f"  meta:  {out_meta}")
    logger.info(f"  info:  {out_info}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
