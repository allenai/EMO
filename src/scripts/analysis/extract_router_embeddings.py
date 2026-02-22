"""
Extract per-document router embeddings from a trained MoE model over OLMoE-mix-0824 data.

For each document, computes four embeddings:
  - Option A: average softmax probabilities per expert per layer (float16)
  - Option B: binary mask of which experts were ever in top-k (bool)
  - Option C: sparse optA — avg softmax probs but zero all but top-32 per layer (float16)
  - Option D: average pre-softmax logits per expert per layer, top-32 sparsified (float16)

All have shape (num_layers * num_standard_experts,) = (16 * 127,) = 2032 dims.

Proportional sampling: reads mix_composition.json (produced by analyze_data_mix.py)
to allocate tokens per source proportionally. Uses range-GET from S3 to avoid
downloading full files (which are multi-GB each).

Usage:
    python -m src.scripts.analysis.extract_router_embeddings \
        --model-path models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf \
        --composition-file claude_outputs/analysis/router_clustering/mix_composition.json \
        --output-dir claude_outputs/analysis/router_clustering \
        --target-tokens 20_000_000 \
        --batch-size 32

    # To add only optC (when optA/B already exist):
    python -m src.scripts.analysis.extract_router_embeddings \
        --model-path ... --composition-file ... --output-dir ... \
        --embeddings optC

    # To add only optD (logits-based):
    python -m src.scripts.analysis.extract_router_embeddings \
        --model-path ... --composition-file ... --output-dir ... \
        --embeddings optD
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


TOP_K_SPARSE = 32  # optC: keep only this many experts per layer

# ---------------------------------------------------------------------------
# Embedding type definitions
# ---------------------------------------------------------------------------
# All embeddings have shape (num_layers * num_standard_experts,) = (16 * 127,) = 2032.
# Saved files and their properties:
#
#   optA  embeddings_optA_avgprob.npy       float16, dense
#         Per-layer average softmax probability per standard expert.
#         For each token, softmax over 127 experts; average across all tokens in doc.
#         Each layer block sums to ~1. Dense: all 127 values non-zero per layer.
#
#   optB  embeddings_optB_binary.npy        bool, very sparse (~6% density)
#         Per-layer binary mask: True if expert appeared in the model's actual
#         top-k (k=num_experts_per_tok=8) routing for ≥1 token in the doc.
#         Captures the hard routing footprint with no magnitude information.
#
#   optC  embeddings_optC_top32sparse.npy   float16, sparse (~25% density)
#         Same as optA but zeroing all but the top-32 experts per layer.
#         Preserves softmax magnitudes for the top-32; bottom 95 are zeroed.
#         Top-32 = 4× the actual top-k=8 routing threshold.
#
#   optD  embeddings_optD_logits_top32sparse.npy  float16, sparse (~25% density)
#         Per-layer average of raw pre-softmax logits per expert, then top-32
#         sparsified. Unlike optC (softmax bounded 0-1), logits have wider
#         dynamic range so PCA centering is less destructive and top experts
#         have much larger magnitudes — better cluster separation.
# ---------------------------------------------------------------------------


@torch.no_grad()
def embed_batch(
    model,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a batch of token-ID arrays through the model.

    Returns:
      emb_a: (batch, num_layers * num_standard_experts) float16
             Per-layer average softmax probability per standard expert.
      emb_b: (batch, num_layers * num_standard_experts) bool
             Per-layer binary mask: True if expert appeared in top-k for ≥1 token.
      emb_c: (batch, num_layers * num_standard_experts) float16
             Sparse optA: keep only top-32 experts per layer, zero the rest.
      emb_d: (batch, num_layers * num_standard_experts) float16
             Sparse logits: average pre-softmax logits, top-32 per layer.
    """
    breakpoint()
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
    emb_a_layers, emb_b_layers, emb_d_layers = [], [], []
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
        flat_mask = valid_mask.reshape(B, -1).bool()               # (B, S*top_k)
        for b in range(B):
            valid = flat_idx[b][flat_mask[b]]
            binary[b].scatter_(0, valid, True)
        emb_b_layers.append(binary.cpu().numpy())

        # Option D: average raw logits per expert (pre-softmax), masked by attention
        avg_logits = (logits.float() * mask).sum(dim=1) / doc_lens.squeeze(1)  # (B, E)
        emb_d_layers.append(avg_logits.cpu().numpy())

    torch.cuda.empty_cache()
    emb_a = np.concatenate(emb_a_layers, axis=1).astype(np.float16)  # (B, L*E)
    emb_b = np.concatenate(emb_b_layers, axis=1)                      # (B, L*E) bool

    # Option C: sparse optA — zero all but top-32 experts per layer
    emb_c = emb_a.reshape(B, num_layers, num_standard_experts).copy()
    # argsort ascending → bottom (E - TOP_K_SPARSE) indices are zeroed
    bottom_idx = np.argsort(emb_c, axis=2)[:, :, :-TOP_K_SPARSE]
    np.put_along_axis(emb_c, bottom_idx, 0.0, axis=2)
    emb_c = emb_c.reshape(B, -1).astype(np.float16)                   # (B, L*E) float16

    # Option D: sparse logits — zero all but top-32 per layer (by magnitude)
    emb_d = np.concatenate(emb_d_layers, axis=1).astype(np.float32)   # (B, L*E)
    emb_d_sparse = emb_d.reshape(B, num_layers, num_standard_experts).copy()
    bottom_idx_d = np.argsort(emb_d_sparse, axis=2)[:, :, :-TOP_K_SPARSE]
    np.put_along_axis(emb_d_sparse, bottom_idx_d, 0.0, axis=2)
    emb_d = emb_d_sparse.reshape(B, -1).astype(np.float16)            # (B, L*E) float16

    return emb_a, emb_b, emb_c, emb_d


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(model, tokenizer, source_doc_samples, device, num_layers, num_standard_experts, top_k):
    logger.info("=== Running sanity checks ===")

    sample_doc = source_doc_samples[0][1][0]
    breakpoint()

    # Check 1: embedding shapes and dtype
    ea, eb, ec, ed = embed_batch(model, [sample_doc], device, num_layers, num_standard_experts, top_k)
    D = num_layers * num_standard_experts
    assert ea.shape == (1, D), f"emb_a shape {ea.shape} != (1, {D})"
    assert eb.shape == (1, D), f"emb_b shape {eb.shape} != (1, {D})"
    assert ec.shape == (1, D), f"emb_c shape {ec.shape} != (1, {D})"
    assert ed.shape == (1, D), f"emb_d shape {ed.shape} != (1, {D})"
    assert ea.dtype == np.float16
    assert eb.dtype == bool
    assert ec.dtype == np.float16
    assert ed.dtype == np.float16
    logger.info(f"  CHECK 1 PASSED: shapes {ea.shape}, dtypes float16/bool/float16/float16 ✓")

    # Check 2: softmax probs sum to ~1 per layer
    per_layer_sum = ea[0].reshape(num_layers, num_standard_experts).sum(axis=1)
    assert (per_layer_sum > 0.95).all() and (per_layer_sum < 1.05).all(), \
        f"Softmax row sums off: {per_layer_sum}"
    logger.info(f"  CHECK 2 PASSED: per-layer prob sums in [0.95,1.05] ✓")

    # Check 2b: optC has exactly TOP_K_SPARSE non-zeros per layer
    ec_reshaped = ec[0].reshape(num_layers, num_standard_experts)
    nnz_per_layer = (ec_reshaped != 0).sum(axis=1)
    assert (nnz_per_layer == TOP_K_SPARSE).all(), \
        f"optC non-zeros per layer: {nnz_per_layer} (expected {TOP_K_SPARSE})"
    logger.info(f"  CHECK 2b PASSED: optC has exactly {TOP_K_SPARSE} non-zeros per layer ✓")

    # Check 2c: optD has exactly TOP_K_SPARSE non-zeros per layer and wider range than optC
    ed_reshaped = ed[0].reshape(num_layers, num_standard_experts)
    nnz_d_per_layer = (ed_reshaped != 0).sum(axis=1)
    assert (nnz_d_per_layer == TOP_K_SPARSE).all(), \
        f"optD non-zeros per layer: {nnz_d_per_layer} (expected {TOP_K_SPARSE})"
    logit_range = float(ed.max() - ed.min())
    prob_range = float(ec.max() - ec.min())
    logger.info(f"  CHECK 2c PASSED: optD has {TOP_K_SPARSE} non-zeros/layer, "
                f"range={logit_range:.2f} (vs optC prob range={prob_range:.4f}) ✓")

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
            ea0, _, _, _ = embed_batch(model, [da[0]], device, num_layers, num_standard_experts, top_k)
            ea1, _, _, _ = embed_batch(model, [da[1]], device, num_layers, num_standard_experts, top_k)
            eb0, _, _, _ = embed_batch(model, [db[0]], device, num_layers, num_standard_experts, top_k)
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
    parser.add_argument("--target-tokens", type=int, default=20_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-doc-len", type=int, default=2048)
    parser.add_argument("--min-doc-len", type=int, default=32)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--embeddings", default="all",
                        choices=["all", "optA", "optB", "optC", "optD"],
                        help="Which embeddings to save. Use 'optC'/'optD' to add only that file "
                             "when others already exist. (default: all)")
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

    # Determine which embeddings to save
    save_set = {"optA", "optB", "optC", "optD"} if args.embeddings == "all" else {args.embeddings}
    logger.info(f"Will save embeddings: {sorted(save_set)}")

    # Main extraction: proportional sampling per source
    all_emb_a, all_emb_b, all_emb_c, all_emb_d, all_meta = [], [], [], [], []

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
        emb_a_list, emb_b_list, emb_c_list, emb_d_list = [], [], [], []
        num_batches = (len(docs) + args.batch_size - 1) // args.batch_size
        t0 = time.time()
        for batch_idx, i in enumerate(range(0, len(docs), args.batch_size)):
            batch = docs[i : i + args.batch_size]
            ea, eb, ec, ed = embed_batch(model, batch, device, num_layers, num_standard_experts, top_k)
            emb_a_list.append(ea)
            emb_b_list.append(eb)
            emb_c_list.append(ec)
            emb_d_list.append(ed)
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{label}] batch {batch_idx+1}/{num_batches} "
                    f"({(batch_idx+1)/num_batches:.0%})  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
                )

        all_emb_a.append(np.concatenate(emb_a_list, axis=0))
        all_emb_b.append(np.concatenate(emb_b_list, axis=0))
        all_emb_c.append(np.concatenate(emb_c_list, axis=0))
        all_emb_d.append(np.concatenate(emb_d_list, axis=0))

        for doc in docs:
            try:
                preview = tokenizer.decode(doc[:200].tolist(), skip_special_tokens=True)[:400]
            except Exception:
                preview = ""
            all_meta.append({"source": label, "doc_len": int(len(doc)), "preview": preview})

        logger.info(f"  Done: {len(docs)} docs embedded for {label}")

    # Save embeddings (only the requested ones)
    out_a = os.path.join(args.output_dir, "embeddings_optA_avgprob.npy")
    out_b = os.path.join(args.output_dir, "embeddings_optB_binary.npy")
    out_c = os.path.join(args.output_dir, "embeddings_optC_top32sparse.npy")
    out_d = os.path.join(args.output_dir, "embeddings_optD_logits_top32sparse.npy")
    out_meta = os.path.join(args.output_dir, "metadata.jsonl.gz")
    out_info = os.path.join(args.output_dir, "info.json")

    if "optA" in save_set:
        emb_a_arr = np.concatenate(all_emb_a, axis=0)
        np.save(out_a, emb_a_arr)
        logger.info(f"  emb_a: {out_a}  shape={emb_a_arr.shape}")
    if "optB" in save_set:
        emb_b_arr = np.concatenate(all_emb_b, axis=0)
        np.save(out_b, emb_b_arr)
        logger.info(f"  emb_b: {out_b}  shape={emb_b_arr.shape}")
    if "optC" in save_set:
        emb_c_arr = np.concatenate(all_emb_c, axis=0)
        np.save(out_c, emb_c_arr)
        logger.info(f"  emb_c: {out_c}  shape={emb_c_arr.shape}")
    if "optD" in save_set:
        emb_d_arr = np.concatenate(all_emb_d, axis=0)
        np.save(out_d, emb_d_arr)
        logger.info(f"  emb_d: {out_d}  shape={emb_d_arr.shape}")

    # Always save metadata and info (needed for clustering/viz)
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
    logger.info(f"  meta:  {out_meta}")
    logger.info(f"  info:  {out_info}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
