"""
Shared utilities for MoE analysis scripts.

Provides S3 data streaming, token parsing, document loading, and model loading
functions used across extract_router_embeddings.py, analyze_expert_coverage.py,
and analyze_weborganizer.py.
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
S3_BASE = "s3://ai2-llm"
ALL_DRESSED_PREFIX = "preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"
BYTES_PER_TOKEN = 4  # headerless raw uint32 binary
EOS_TOKEN_ID = 100257


# ── S3 helpers ───────────────────────────────────────────────────────────────

def s3_ls(prefix: str) -> List[str]:
    """List immediate children of an S3 prefix (directories and files)."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://ai2-llm/{prefix}/"],
        capture_output=True, text=True,
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
    """
    List .npy files in a topic/vigintile directory.
    Returns list of (s3_path, size_bytes).
    """
    prefix = f"{ALL_DRESSED_PREFIX}/{topic}/{vigintile}"
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://ai2-llm/{prefix}/"],
        capture_output=True, text=True,
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


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    """Load an HF MoE model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def get_moe_config(model) -> Dict[str, int]:
    """
    Extract MoE-related config from a loaded model.

    Returns dict with keys: num_layers, num_experts, num_shared_experts,
    num_standard_experts, top_k, routed_top_k, emb_dim.

    top_k is the total number of experts per token (including shared).
    routed_top_k = top_k - num_shared_experts is the number of experts
    selected by the router (excluding always-on shared experts).
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_all_experts = cfg.num_experts if hasattr(cfg, "num_experts") else cfg.num_local_experts
    num_shared = getattr(cfg, "num_shared_experts", 0)
    num_standard_experts = num_all_experts - num_shared
    top_k = cfg.num_experts_per_tok
    routed_top_k = top_k - num_shared
    emb_dim = num_layers * num_standard_experts

    assert num_shared < top_k, \
        f"num_shared_experts ({num_shared}) must be less than top_k ({top_k})"

    logger.info(f"Model: {num_layers} layers, {num_standard_experts} standard experts "
                f"({num_shared} shared), top_k={top_k} (routed={routed_top_k}), "
                f"emb_dim={emb_dim}")

    return {
        "num_layers": num_layers,
        "num_experts": num_all_experts,
        "num_shared_experts": num_shared,
        "num_standard_experts": num_standard_experts,
        "top_k": top_k,
        "routed_top_k": routed_top_k,
        "emb_dim": emb_dim,
    }
