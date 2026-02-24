"""
Analyze expert coverage across weborganizer topics.

Samples tokens uniformly across cc_all_dressed topics (20M tokens by default,
equally distributed per topic) and processes documents through an MoE model
to analyze how experts cover different topic domains.

This script reuses the mix_composition.json from analyze_weborganizer.py
(or generates one on the fly if not found). It loads documents from S3,
runs them through the model, and collects per-document expert activation
statistics.

Usage:
    python -u -m src.scripts.analysis.analyze_expert_coverage \
        --model-path models/.../step30995-hf \
        --output-dir claude_outputs/analysis/expert_coverage_weborganizer \
        --target-tokens 20_000_000 \
        --batch-size 32

    # If mix_composition.json already exists (e.g. from analyze_weborganizer.py):
    python -u -m src.scripts.analysis.analyze_expert_coverage \
        --model-path models/.../step30995-hf \
        --composition-file claude_outputs/analysis/router_clustering_weborganizer/mix_composition.json \
        --output-dir claude_outputs/analysis/expert_coverage_weborganizer \
        --target-tokens 20_000_000 \
        --batch-size 32
"""

import argparse
import gzip
import json
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from src.scripts.analysis.utils import (
    ALL_DRESSED_PREFIX,
    BYTES_PER_TOKEN,
    EOS_TOKEN_ID,
    get_moe_config,
    list_npy_files,
    load_model_and_tokenizer,
    load_source_documents,
    s3_ls,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Composition helpers ──────────────────────────────────────────────────────

def generate_uniform_composition(output_dir: str) -> Dict[str, Any]:
    """
    Discover weborganizer topics on S3 and build a uniform mix_composition.json.
    Equivalent to running analyze_weborganizer.py but integrated here for convenience.
    """
    logger.info("Discovering topics on S3 ...")
    topics = sorted(s3_ls(ALL_DRESSED_PREFIX))
    logger.info(f"Found {len(topics)} topics: {topics}")

    topic_files: Dict[str, list] = {}
    topic_vigintile: Dict[str, str] = {}

    for topic in topics[:2]:
        vigs = sorted(e for e in s3_ls(f"{ALL_DRESSED_PREFIX}/{topic}") if e.startswith("vigintile_"))
        if not vigs:
            logger.warning(f"  [{topic}] no vigintiles, skipping")
            continue
        max_vig = vigs[-1]
        topic_vigintile[topic] = max_vig

        files = list_npy_files(topic, max_vig)
        if not files:
            logger.warning(f"  [{topic}] no .npy files in {max_vig}, skipping")
            continue
        topic_files[topic] = files
        total_bytes = sum(sz for _, sz in files)
        logger.info(f"  [{topic}] {max_vig}: {len(files)} files, ~{total_bytes // BYTES_PER_TOKEN:,} tokens")

    valid_topics = sorted(topic_files.keys())
    num_topics = len(valid_topics)
    uniform_fraction = 1.0 / num_topics

    composition = {
        "description": "cc_all_dressed uniform mixing across topics (for expert coverage analysis)",
        "num_topics": num_topics,
        "sources": {
            topic: {
                "vigintile": topic_vigintile[topic],
                "num_files": len(topic_files[topic]),
                "est_tokens": sum(sz for _, sz in topic_files[topic]) // BYTES_PER_TOKEN,
                "total_bytes": sum(sz for _, sz in topic_files[topic]),
                "fraction": round(uniform_fraction, 6),
                "all_files": [path for path, _ in topic_files[topic]],
            }
            for topic in valid_topics
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    comp_path = os.path.join(output_dir, "mix_composition.json")
    with open(comp_path, "w") as f:
        json.dump(composition, f, indent=2)
    logger.info(f"Saved mix_composition.json -> {comp_path}")
    return composition


# ── Model inference ──────────────────────────────────────────────────────────

TOP_K = 8  # number of experts selected per token per layer


@torch.no_grad()
def process_batch(
    model: Any,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
):
    """
    Run a batch of documents through the model, select top-8 experts per token
    per layer, and return per-document expert activation counts.

    Returns:
        counts: np.ndarray of shape (B, num_layers * num_standard_experts), dtype int32.
                Each entry is the number of tokens in that document that selected
                that expert (via top-8) at that layer.
        doc_lens: np.ndarray of shape (B,), the number of real (non-padding) tokens
                  per document.
    """
    B = len(batch_docs)
    max_len = max(len(doc) for doc in batch_docs)

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

    # Accumulate counts: (B, num_layers, num_standard_experts)
    counts = torch.zeros(B, num_layers, num_standard_experts, dtype=torch.int32, device=device)
    mask = attention_mask.unsqueeze(-1)  # (B, S, 1)

    breakpoint()

    for layer_idx, layer_logits in enumerate(outputs.router_logits):
        # layer_logits: (B*S, E_total) -> (B, S, E_standard)
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts]

        # Top-8 expert indices per token: (B, S, TOP_K)
        top_indices = logits.topk(TOP_K, dim=-1).indices

        # Create one-hot and mask out padding, then sum across tokens
        # one_hot: (B, S, TOP_K, E) -> sum over S and TOP_K dims
        one_hot = torch.zeros(B, max_len, TOP_K, num_standard_experts,
                              dtype=torch.int32, device=device)
        one_hot.scatter_(3, top_indices.unsqueeze(-1), 1)

        # Mask padding tokens: mask is (B, S, 1), broadcast over TOP_K and E
        one_hot = one_hot * mask.unsqueeze(2)  # (B, S, TOP_K, E)

        # Sum over tokens and top-k selections: (B, E)
        counts[:, layer_idx, :] = one_hot.sum(dim=(1, 2))

    # Flatten to (B, num_layers * num_standard_experts)
    counts_flat = counts.view(B, -1).cpu().numpy()
    doc_lens = attention_mask.sum(dim=1).cpu().numpy()

    torch.cuda.empty_cache()
    return counts_flat, doc_lens


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert coverage across weborganizer topics"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HF MoE model checkpoint")
    parser.add_argument("--composition-file", type=str, default=None,
                        help="Path to existing mix_composition.json. "
                             "If not provided, will discover topics on S3.")
    parser.add_argument("--output-dir", type=str,
                        default="claude_outputs/analysis/expert_coverage_weborganizer",
                        help="Output directory for results")
    parser.add_argument("--target-tokens", type=int, default=20_000_000,
                        help="Total tokens to sample (distributed uniformly across topics)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for model inference")
    parser.add_argument("--min-doc-len", type=int, default=32)
    parser.add_argument("--max-doc-len", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load or generate mix composition ─────────────────────────────
    if args.composition_file and os.path.exists(args.composition_file):
        logger.info(f"Loading composition from {args.composition_file}")
        with open(args.composition_file) as f:
            composition = json.load(f)
    else:
        logger.info("No composition file provided; discovering topics on S3 ...")
        composition = generate_uniform_composition(args.output_dir)

    sources = composition["sources"]
    num_topics = len(sources)
    tokens_per_topic = args.target_tokens // num_topics
    logger.info(f"{num_topics} topics, {tokens_per_topic:,} tokens per topic "
                f"({args.target_tokens:,} total)")

    # ── Step 2: Load documents from each topic ───────────────────────────────
    logger.info("\nLoading documents from S3 ...")
    all_docs: List[np.ndarray] = []
    all_labels: List[str] = []

    for topic, info in sorted(sources.items()):
        s3_files = info["all_files"]
        logger.info(f"  [{topic}] loading ~{tokens_per_topic:,} tokens ...")
        docs = load_source_documents(
            s3_files, tokens_per_topic, args.min_doc_len, args.max_doc_len
        )
        actual_tokens = sum(len(d) for d in docs)
        logger.info(f"  [{topic}] got {len(docs)} docs, {actual_tokens:,} tokens")
        all_docs.extend(docs)
        all_labels.extend([topic] * len(docs))

    total_tokens = sum(len(d) for d in all_docs)
    logger.info(f"\nTotal: {len(all_docs)} docs, {total_tokens:,} tokens "
                f"across {num_topics} topics")

    # ── Step 3: Load model ───────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = str(next(model.parameters()).device)
    moe_cfg = get_moe_config(model)
    num_layers = moe_cfg["num_layers"]
    num_standard_experts = moe_cfg["num_standard_experts"]

    # ── Step 4: Process documents through model ──────────────────────────────
    logger.info("\nProcessing documents through model ...")
    logger.info(f"  Top-{TOP_K} expert selection per token per layer")

    num_docs = len(all_docs)
    num_batches = (num_docs + args.batch_size - 1) // args.batch_size
    emb_dim = num_layers * num_standard_experts

    # Collect raw counts: (num_docs, num_layers * num_standard_experts)
    all_counts = np.zeros((num_docs, emb_dim), dtype=np.int32)
    all_doc_lens = np.zeros(num_docs, dtype=np.int32)
    breakpoint()
    t0 = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, num_docs)
        batch_docs = all_docs[batch_start:batch_end]

        batch_counts, batch_doc_lens = process_batch(
            model, batch_docs, device, num_layers, num_standard_experts
        )

        all_counts[batch_start:batch_end] = batch_counts
        all_doc_lens[batch_start:batch_end] = batch_doc_lens

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"  Batch {batch_idx+1}/{num_batches} "
                f"({(batch_idx+1)/num_batches:.0%})  "
                f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
            )

    # ── Step 4b: Validate counts ─────────────────────────────────────────────
    logger.info("\nValidating counts ...")
    assert all_counts.min() >= 0, f"Counts have negative values: min={all_counts.min()}"

    # Per-layer counts for each doc should sum to doc_len * TOP_K
    # (each token selects TOP_K experts per layer)
    counts_per_layer = all_counts.reshape(num_docs, num_layers, num_standard_experts)
    layer_sums = counts_per_layer.sum(axis=2)  # (num_docs, num_layers)
    expected_sums = all_doc_lens[:, None] * TOP_K  # (num_docs, 1)
    assert (layer_sums == expected_sums).all(), \
        f"Per-layer count sums don't match doc_len * top_k. " \
        f"Got range [{layer_sums.min()}, {layer_sums.max()}], " \
        f"expected range [{expected_sums.min()}, {expected_sums.max()}]"

    # Each individual count should be <= doc_len (at most every token chose that expert)
    max_per_doc = all_counts.max(axis=1)  # (num_docs,)
    assert (max_per_doc <= all_doc_lens).all(), \
        f"Some expert counts exceed doc length: max_count={max_per_doc.max()}, " \
        f"max_doc_len={all_doc_lens.max()}"

    logger.info(f"  All counts >= 0: PASSED")
    logger.info(f"  All counts <= doc_len: PASSED (max={max_per_doc.max()}, "
                f"max_doc_len={all_doc_lens.max()})")
    logger.info(f"  Per-layer sums == doc_len * {TOP_K}: PASSED")
    logger.info(f"  Counts shape: {all_counts.shape}")
    logger.info(f"  Counts range: [{all_counts.min()}, {all_counts.max()}]")

    # ── Step 4c: Normalize by number of tokens ───────────────────────────────
    # Divide each document's counts by its token count to get frequencies
    coverage_freq = all_counts.astype(np.float32) / all_doc_lens[:, None].astype(np.float32)
    logger.info(f"  Coverage freq shape: {coverage_freq.shape}")
    logger.info(f"  Coverage freq range: [{coverage_freq.min():.4f}, {coverage_freq.max():.4f}]")
    # Each layer's frequencies should sum to TOP_K
    freq_per_layer = coverage_freq.reshape(num_docs, num_layers, num_standard_experts)
    layer_freq_sums = freq_per_layer.sum(axis=2)
    logger.info(f"  Per-layer freq sums: [{layer_freq_sums.min():.4f}, {layer_freq_sums.max():.4f}] "
                f"(expected {TOP_K})")

    # ── Step 5: Save results ─────────────────────────────────────────────────
    # Save raw counts
    counts_path = os.path.join(args.output_dir, "expert_counts.npy")
    np.save(counts_path, all_counts)
    logger.info(f"Saved raw counts -> {counts_path}  shape={all_counts.shape}")

    # Save normalized frequencies
    freq_path = os.path.join(args.output_dir, "expert_freq.npy")
    np.save(freq_path, coverage_freq)
    logger.info(f"Saved frequencies -> {freq_path}  shape={coverage_freq.shape}")

    # Save metadata
    metadata_path = os.path.join(args.output_dir, "metadata.jsonl.gz")
    with gzip.open(metadata_path, "wt") as f:
        for i, (doc, label) in enumerate(zip(all_docs, all_labels)):
            record = {
                "doc_idx": i,
                "source": label,
                "doc_len": int(len(doc)),
                "preview": tokenizer.decode(doc[:100].tolist(), skip_special_tokens=True)[:200],
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved metadata -> {metadata_path}")

    # Save info
    info = {
        "model_path": args.model_path,
        "num_layers": num_layers,
        "num_standard_experts": num_standard_experts,
        "top_k": TOP_K,
        "num_topics": num_topics,
        "target_tokens": args.target_tokens,
        "tokens_per_topic": tokens_per_topic,
        "total_docs": num_docs,
        "total_tokens": int(total_tokens),
        "topics": sorted(sources.keys()),
    }
    info_path = os.path.join(args.output_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved info -> {info_path}")

    logger.info("\nDone! Expert coverage analysis complete.")


if __name__ == "__main__":
    main()
