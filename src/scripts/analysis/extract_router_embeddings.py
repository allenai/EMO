"""
Extract per-document router embeddings from a trained MoE model over OLMoE-mix-0824 data.

For each document, computes two embeddings:
  - Option A: average softmax probabilities per expert per layer (float16)
  - Option B: binary mask of which experts were ever in top-k (bool)

Both have shape (num_layers * num_standard_experts,) = (16 * 127,) = 2032 dims.

Proportional sampling: reads mix_composition.json (produced by analyze_data_mix.py)
to allocate tokens per source proportionally. Uses range-GET from S3 to avoid
downloading full files (which are multi-GB each).

Usage:
    python -m src.scripts.analysis.extract_router_embeddings \
        --model-path models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf \
        --composition-file claude_outputs/analysis/router_clustering/mix_composition.json \
        --output-dir claude_outputs/analysis/router_clustering \
        --target-tokens 500_000_000 \
        --batch-size 8
"""

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 100257
BYTES_PER_TOKEN = 4  # uint32


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
    # Add 50% headroom since not every byte becomes a valid document
    # (some are split at boundaries, some too short/long)
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
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a batch of token-ID arrays through the model.

    Returns:
      emb_a: (batch, num_layers * num_standard_experts) float16
             Per-layer average softmax probability per standard expert.
      emb_b: (batch, num_layers * num_standard_experts) bool
             Per-layer binary mask: True if expert appeared in top-k for ≥1 token.
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

    # router_logits: tuple of (B*S, num_all_experts) per layer
    emb_a_layers, emb_b_layers = [], []
    for layer_logits in outputs.router_logits:
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts]  # (B, S, E)

        # Option A: masked average softmax
        probs = F.softmax(logits.float(), dim=-1).half()          # (B, S, E)
        mask = attention_mask.unsqueeze(-1)                        # (B, S, 1)
        doc_lens = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).half()  # (B, 1, 1)
        avg_probs = (probs * mask).sum(dim=1) / doc_lens.squeeze(1)  # (B, E)
        emb_a_layers.append(avg_probs.cpu().numpy())

        # Option B: binary — which experts appeared in top-k for ≥1 non-padding token
        topk_idx = torch.topk(logits, k=top_k, dim=-1).indices   # (B, S, top_k)
        binary = torch.zeros(B, num_standard_experts, dtype=torch.bool, device=device)
        valid_mask = attention_mask.unsqueeze(-1).expand_as(topk_idx)  # (B, S, top_k)
        flat_idx = topk_idx.reshape(B, -1)                         # (B, S*top_k)
        flat_mask = valid_mask.reshape(B, -1)                      # (B, S*top_k)
        for b in range(B):
            valid = flat_idx[b][flat_mask[b]]
            binary[b].scatter_(0, valid, True)
        emb_b_layers.append(binary.cpu().numpy())

    torch.cuda.empty_cache()
    emb_a = np.concatenate(emb_a_layers, axis=1).astype(np.float16)  # (B, L*E)
    emb_b = np.concatenate(emb_b_layers, axis=1)                      # (B, L*E) bool
    return emb_a, emb_b


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(model, tokenizer, source_doc_samples, device, num_layers, num_standard_experts, top_k):
    logger.info("=== Running sanity checks ===")

    sample_doc = source_doc_samples[0][1][0]

    # Check 1: embedding shapes and dtype
    ea, eb = embed_batch(model, [sample_doc], device, num_layers, num_standard_experts, top_k)
    D = num_layers * num_standard_experts
    assert ea.shape == (1, D), f"emb_a shape {ea.shape} != (1, {D})"
    assert eb.shape == (1, D), f"emb_b shape {eb.shape} != (1, {D})"
    assert ea.dtype == np.float16
    assert eb.dtype == bool
    logger.info(f"  CHECK 1 PASSED: shapes {ea.shape}, dtypes float16/bool ✓")

    # Check 2: softmax probs sum to ~1 per layer
    per_layer_sum = ea[0].reshape(num_layers, num_standard_experts).sum(axis=1)
    assert (per_layer_sum > 0.95).all() and (per_layer_sum < 1.05).all(), \
        f"Softmax row sums off: {per_layer_sum}"
    logger.info(f"  CHECK 2 PASSED: per-layer prob sums in [0.95,1.05] ✓")

    # Check 3: decoded text looks sane
    logger.info("  CHECK 3: decoding sample docs per source")
    for label, docs in source_doc_samples[:3]:
        decoded = tokenizer.decode(docs[0][:80].tolist(), skip_special_tokens=True)
        logger.info(f"    [{label}] {decoded[:120]!r}")

    # Check 4: same-source docs more similar than cross-source (option A)
    if len(source_doc_samples) >= 2:
        def cos(x, y):
            x, y = x.astype(np.float32), y.astype(np.float32)
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

        la, da = source_doc_samples[0]
        lb, db = source_doc_samples[1]
        if len(da) >= 2 and len(db) >= 1:
            ea0, _ = embed_batch(model, [da[0]], device, num_layers, num_standard_experts, top_k)
            ea1, _ = embed_batch(model, [da[1]], device, num_layers, num_standard_experts, top_k)
            eb0, _ = embed_batch(model, [db[0]], device, num_layers, num_standard_experts, top_k)
            same = cos(ea0[0], ea1[0])
            cross = cos(ea0[0], eb0[0])
            logger.info(f"  CHECK 4: same-source ({la}) cos={same:.3f}  cross-source ({la} vs {lb}) cos={cross:.3f}")
            if same > cross:
                logger.info("  CHECK 4 PASSED: same-source more similar ✓")
            else:
                logger.info("  CHECK 4 NOTE: cross-source similarity not lower — may be fine for similar domains")

    logger.info("=== Sanity checks complete ===\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--composition-file", required=True,
                        help="mix_composition.json from analyze_data_mix.py")
    parser.add_argument("--output-dir", default="claude_outputs/analysis/router_clustering")
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-doc-len", type=int, default=2048)
    parser.add_argument("--min-doc-len", type=int, default=32)
    parser.add_argument("--sanity-check-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load proportional composition
    with open(args.composition_file) as f:
        composition = json.load(f)

    sources = composition["sources"]  # {label: {fraction, all_files, ...}}
    logger.info(f"Loaded composition for {len(sources)} sources")
    for label, info in sorted(sources.items(), key=lambda x: -x[1]["fraction"]):
        alloc = int(info["fraction"] * args.target_tokens)
        logger.info(f"  {label}: {info['fraction']:.2%} → {alloc:,} tokens")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = str(next(model.parameters()).device)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_all_experts = cfg.num_experts if hasattr(cfg, "num_experts") else cfg.num_local_experts
    num_shared = getattr(cfg, "num_shared_experts", 0)
    num_standard_experts = num_all_experts - num_shared
    top_k = cfg.num_experts_per_tok
    emb_dim = num_layers * num_standard_experts

    logger.info(f"Model: {num_layers} layers, {num_standard_experts} standard experts, top-{top_k}")
    logger.info(f"Embedding dim: {emb_dim}")

    # Collect a few docs per source for sanity checks
    source_doc_samples = []
    for label, info in sources.items():
        raw = stream_bytes_from_s3(info["all_files"][0], 3_000_000)
        toks = tokens_from_bytes(raw)
        docs = list(iter_documents(toks, args.min_doc_len, args.max_doc_len))[:5]
        if docs:
            source_doc_samples.append((label, docs))

    run_sanity_checks(model, tokenizer, source_doc_samples, device, num_layers, num_standard_experts, top_k)

    if args.sanity_check_only:
        logger.info("--sanity-check-only specified, exiting.")
        return

    # Main extraction: proportional sampling per source
    all_emb_a, all_emb_b, all_meta = [], [], []

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
        import time
        emb_a_list, emb_b_list = [], []
        num_batches = (len(docs) + args.batch_size - 1) // args.batch_size
        t0 = time.time()
        for batch_idx, i in enumerate(range(0, len(docs), args.batch_size)):
            batch = docs[i : i + args.batch_size]
            ea, eb = embed_batch(model, batch, device, num_layers, num_standard_experts, top_k)
            emb_a_list.append(ea)
            emb_b_list.append(eb)
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{label}] batch {batch_idx+1}/{num_batches} "
                    f"({(batch_idx+1)/num_batches:.0%})  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
                )

        source_emb_a = np.concatenate(emb_a_list, axis=0)
        source_emb_b = np.concatenate(emb_b_list, axis=0)
        all_emb_a.append(source_emb_a)
        all_emb_b.append(source_emb_b)

        for doc in docs:
            try:
                preview = tokenizer.decode(doc[:200].tolist(), skip_special_tokens=True)[:400]
            except Exception:
                preview = ""
            all_meta.append({"source": label, "doc_len": int(len(doc)), "preview": preview})

        logger.info(f"  Done: {len(docs)} docs embedded for {label}")

    # Save
    emb_a_arr = np.concatenate(all_emb_a, axis=0)
    emb_b_arr = np.concatenate(all_emb_b, axis=0)

    out_a = os.path.join(args.output_dir, "embeddings_optA_avgprob.npy")
    out_b = os.path.join(args.output_dir, "embeddings_optB_binary.npy")
    out_meta = os.path.join(args.output_dir, "metadata.jsonl.gz")
    out_info = os.path.join(args.output_dir, "info.json")

    np.save(out_a, emb_a_arr)
    np.save(out_b, emb_b_arr)

    with gzip.open(out_meta, "wt") as f:
        for m in all_meta:
            f.write(json.dumps(m) + "\n")

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
        "source_doc_counts": dict(source_counts),
    }
    with open(out_info, "w") as f:
        json.dump(info_out, f, indent=2)

    logger.info(f"\nSaved {len(all_meta)} document embeddings")
    logger.info(f"  emb_a: {out_a}  shape={emb_a_arr.shape}")
    logger.info(f"  emb_b: {out_b}  shape={emb_b_arr.shape}")
    logger.info(f"  meta:  {out_meta}")
    logger.info(f"  info:  {out_info}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
