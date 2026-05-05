"""
Shared utilities for the clustering pipeline.

Provides S3 data streaming, token parsing, document loading, and model loading.
"""

import logging
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
S3_BASE = "s3://ai2-llm"
ALL_DRESSED_PREFIX = (
    "preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"
)
BYTES_PER_TOKEN = 4  # headerless raw uint32 binary
EOS_TOKEN_ID = 100257


# ── S3 helpers ───────────────────────────────────────────────────────────────


def s3_ls(prefix: str) -> List[str]:
    """List immediate children of an S3 prefix (directories and files)."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://ai2-llm/{prefix}/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 ls failed for {prefix}: {result.stderr[:200]}")
    entries = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if line.strip().startswith("PRE"):
            entries.append(parts[1].rstrip("/"))
        elif len(parts) == 4:
            entries.append(parts[3])
    return entries


def list_npy_files(topic: str, vigintile: str) -> List[Tuple[str, int]]:
    """List .npy files in a topic/vigintile directory. Returns [(s3_path, size_bytes)]."""
    prefix = f"{ALL_DRESSED_PREFIX}/{topic}/{vigintile}"
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://ai2-llm/{prefix}/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Could not list {prefix}: {result.stderr[:100]}")
        return []

    files = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            fname, size_str = parts[3], parts[2]
            if fname.endswith(".npy"):
                try:
                    s3_path = f"s3://ai2-llm/{prefix}/{fname}"
                    files.append((s3_path, int(size_str)))
                except ValueError:
                    pass
    return sorted(files, key=lambda x: x[0])


def stream_bytes_from_s3(s3_path: str, num_bytes: int) -> bytes:
    """Range-GET the first `num_bytes` of an S3 file into memory."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                "ai2-llm",
                "--key",
                s3_path.replace("s3://ai2-llm/", ""),
                "--range",
                f"bytes=0-{num_bytes - 1}",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"S3 range-GET failed for {s3_path}: {result.stderr[:200]}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def stream_bytes_from_s3_range(s3_path: str, offset: int, num_bytes: int) -> bytes:
    """Range-GET `num_bytes` starting at `offset` from an S3 file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    try:
        end = offset + num_bytes - 1
        result = subprocess.run(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                "ai2-llm",
                "--key",
                s3_path.replace("s3://ai2-llm/", ""),
                "--range",
                f"bytes={offset}-{end}",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"S3 range-GET failed for {s3_path} [{offset}-{end}]: {result.stderr[:200]}"
            )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_s3_file_sizes(s3_paths: List[str]) -> Dict[str, int]:
    """Query S3 for file sizes, batching by directory prefix."""
    prefix_to_files: Dict[str, List[str]] = defaultdict(list)
    for path in s3_paths:
        prefix = path.rsplit("/", 1)[0] + "/"
        prefix_to_files[prefix].append(path)

    sizes: Dict[str, int] = {}
    for prefix, files in prefix_to_files.items():
        logger.info(f"  Querying file sizes in {prefix.split('/')[-2]}/ ...")
        result = subprocess.run(
            ["aws", "s3", "ls", prefix],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"  Could not list {prefix}: {result.stderr[:100]}")
            continue
        file_sizes: Dict[str, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                try:
                    file_sizes[parts[3]] = int(parts[2])
                except ValueError:
                    pass
        for s3_path in files:
            fname = s3_path.rsplit("/", 1)[-1]
            if fname in file_sizes:
                sizes[s3_path] = file_sizes[fname]
    return sizes


# ── Token parsing ────────────────────────────────────────────────────────────


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
    min_doc_len: int = 32,
    max_doc_len: int = 2048,
) -> List[np.ndarray]:
    """Stream documents from S3 sequentially until target_tokens is reached."""
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


def load_source_documents_shuffled(
    s3_files: List[str],
    target_tokens: int,
    min_doc_len: int = 32,
    max_doc_len: int = 2048,
    chunk_bytes: int = 2_000_000,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Randomly sample documents from across all S3 files for a single source.

    Picks random byte offsets across files (weighted by size) so that documents
    are sampled uniformly from the entire dataset. Skips partial first documents
    at each offset.
    """
    rng = random.Random(seed)

    logger.info(f"  Querying file sizes for {len(s3_files)} files ...")
    file_sizes = get_s3_file_sizes(s3_files)

    pool = [(path, size) for path, size in file_sizes.items() if size > chunk_bytes]
    if not pool:
        pool = [(path, size) for path, size in file_sizes.items() if size > 0]
    if not pool:
        logger.warning("  No files with known sizes, falling back to sequential loading")
        return load_source_documents(s3_files, target_tokens, min_doc_len, max_doc_len)

    total_pool_bytes = sum(size for _, size in pool)
    weights = [size / total_pool_bytes for _, size in pool]
    logger.info(f"  File pool: {len(pool)} files, {total_pool_bytes / 1e9:.1f} GB total")

    docs = []
    collected_tokens = 0
    attempts = 0
    max_attempts = 200

    while collected_tokens < target_tokens and attempts < max_attempts:
        attempts += 1

        (s3_path, file_size) = rng.choices(pool, weights=weights, k=1)[0]

        max_offset = max(0, file_size - chunk_bytes)
        byte_offset = rng.randint(0, max_offset) & ~3  # align to 4 bytes

        logger.info(
            f"  [{attempts}] Sampling {chunk_bytes / 1e6:.1f} MB from "
            f"{s3_path.split('/')[-1]} offset={byte_offset:,} ..."
        )
        try:
            raw = stream_bytes_from_s3_range(s3_path, byte_offset, chunk_bytes)
        except Exception as e:
            logger.warning(f"  Skipping chunk: {e}")
            continue

        tokens = tokens_from_bytes(raw)
        if len(tokens) == 0:
            continue

        chunk_docs = list(iter_documents(tokens, min_doc_len, max_doc_len))

        # Skip partial first document if we started mid-file
        if byte_offset > 0 and len(chunk_docs) > 0:
            chunk_docs = chunk_docs[1:]

        for doc in chunk_docs:
            docs.append(doc.copy())
            collected_tokens += len(doc)
            if collected_tokens >= target_tokens:
                break

    logger.info(
        f"  Collected {len(docs)} docs / {collected_tokens:,} tokens " f"({attempts} random chunks)"
    )
    return docs


# ── Model loading ────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_path: str):
    """Load an HF MoE model and tokenizer.

    `trust_remote_code=True` is required for the released HF Hub checkpoints
    (allenai/Emo_*, allenai/StdMoE_*) which ship custom modeling code.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_moe_config(model) -> Dict[str, int]:
    """
    Extract MoE config from a loaded model.

    Returns dict with: num_layers, num_experts, num_shared_experts,
    num_standard_experts, top_k, routed_top_k, emb_dim.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_all_experts = cfg.num_experts if hasattr(cfg, "num_experts") else cfg.num_local_experts
    num_shared = getattr(cfg, "num_shared_experts", 0)
    num_standard_experts = num_all_experts - num_shared
    top_k = cfg.num_experts_per_tok
    routed_top_k = top_k - num_shared
    emb_dim = num_layers * num_standard_experts

    assert (
        num_shared < top_k
    ), f"num_shared_experts ({num_shared}) must be less than top_k ({top_k})"

    logger.info(
        f"Model: {num_layers} layers, {num_standard_experts} standard experts "
        f"({num_shared} shared), top_k={top_k} (routed={routed_top_k}), "
        f"emb_dim={emb_dim}"
    )

    return {
        "num_layers": num_layers,
        "num_experts": num_all_experts,
        "num_shared_experts": num_shared,
        "num_standard_experts": num_standard_experts,
        "top_k": top_k,
        "routed_top_k": routed_top_k,
        "emb_dim": emb_dim,
    }


# ── Prompt tokenization ─────────────────────────────────────────────────────


def tokenize_prompts(
    prompts: List[str],
    tokenizer,
    max_length: int = 2048,
) -> List[np.ndarray]:
    """Tokenize prompt strings into numpy arrays of token IDs."""
    docs = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        if len(token_ids) > 0:
            docs.append(np.array(token_ids, dtype=np.int32))
    return docs
