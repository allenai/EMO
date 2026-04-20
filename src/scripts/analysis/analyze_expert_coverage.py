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
from scipy.stats import entropy as scipy_entropy

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

    for topic in topics:
        vigs = sorted(
            e for e in s3_ls(f"{ALL_DRESSED_PREFIX}/{topic}") if e.startswith("vigintile_")
        )
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
        logger.info(
            f"  [{topic}] {max_vig}: {len(files)} files, ~{total_bytes // BYTES_PER_TOKEN:,} tokens"
        )

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


@torch.no_grad()
def process_batch(
    model: Any,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
    routed_top_k: int,
) -> np.ndarray:
    """
    Run a batch of documents through the model, select top routed_top_k experts
    per token per layer, and return per-document normalized expert activation
    frequencies.

    routed_top_k is the number of experts selected by the router, excluding
    shared experts (i.e. num_experts_per_tok - num_shared_experts).

    For each document, counts how many tokens selected each expert at each
    layer, then divides by the document length. Values are in [0, 1] and each
    layer's frequencies sum to routed_top_k.

    Returns:
        freq: np.ndarray of shape (B, num_layers * num_standard_experts), dtype float32.
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

    for layer_idx, layer_logits in enumerate(outputs.router_logits):
        # layer_logits: (B*S, E_total) -> (B, S, E_standard)
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts]

        # Top routed_top_k expert indices per token: (B, S, routed_top_k)
        top_indices = logits.topk(routed_top_k, dim=-1).indices

        # Flatten top-k indices to (B, S*routed_top_k) and use scatter_add_ to count
        flat_indices = top_indices.reshape(B, -1)  # (B, S*routed_top_k)

        # Build a mask for valid (non-padding) tokens, repeated for each top-k slot
        valid = attention_mask.unsqueeze(-1).expand(-1, -1, routed_top_k).reshape(B, -1)

        # scatter_add_ ones at expert indices, masked by valid tokens
        counts[:, layer_idx, :].scatter_add_(1, flat_indices, valid.to(torch.int32))

    # Normalize: divide by doc_len to get frequencies in [0, 1]
    doc_lens = attention_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
    freq = counts.view(B, -1).float() / doc_lens  # (B, num_layers * num_standard_experts)

    torch.cuda.empty_cache()
    return freq.cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert coverage across weborganizer topics"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to HF MoE model checkpoint"
    )
    parser.add_argument(
        "--composition-file",
        type=str,
        default=None,
        help="Path to existing mix_composition.json. "
        "If not provided, will discover topics on S3.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. If not provided, "
        "auto-derived as expert_coverage_weborganizer/<model_name>/",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=20_000_000,
        help="Total tokens to sample (distributed uniformly across topics)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference")
    parser.add_argument("--min-doc-len", type=int, default=32)
    parser.add_argument("--max-doc-len", type=int, default=2048)
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: only load from the first 2 data sources"
    )
    args = parser.parse_args()

    # Derive output dir from model path if not specified
    if args.output_dir is None:
        # Extract model name: strip trailing /step*-hf, take last path component
        model_name = args.model_path.rstrip("/")
        # Remove step*-hf suffix if present
        parts = model_name.split("/")
        if parts[-1].startswith("step") and parts[-1].endswith("-hf"):
            model_name = parts[-2]
        else:
            model_name = parts[-1]
        args.output_dir = os.path.join(
            "claude_outputs/analysis/expert_coverage_weborganizer", model_name
        )
        logger.info(f"Auto-derived output dir: {args.output_dir}")

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
    if args.debug:
        # Only keep the first 2 sources for fast iteration
        kept = dict(list(sorted(sources.items()))[:2])
        logger.info(f"DEBUG mode: using 2/{len(sources)} sources: {list(kept.keys())}")
        sources = kept
    num_topics = len(sources)
    tokens_per_topic = args.target_tokens // num_topics
    logger.info(
        f"{num_topics} topics, {tokens_per_topic:,} tokens per topic "
        f"({args.target_tokens:,} total)"
    )

    # ── Step 2: Load documents from each topic ───────────────────────────────
    logger.info("\nLoading documents from S3 ...")
    all_docs: List[np.ndarray] = []
    all_labels: List[str] = []

    for topic, info in sorted(sources.items()):
        s3_files = info["all_files"]
        logger.info(f"  [{topic}] loading ~{tokens_per_topic:,} tokens ...")
        docs = load_source_documents(s3_files, tokens_per_topic, args.min_doc_len, args.max_doc_len)
        actual_tokens = sum(len(d) for d in docs)
        logger.info(f"  [{topic}] got {len(docs)} docs, {actual_tokens:,} tokens")
        all_docs.extend(docs)
        all_labels.extend([topic] * len(docs))

    total_tokens = sum(len(d) for d in all_docs)
    logger.info(
        f"\nTotal: {len(all_docs)} docs, {total_tokens:,} tokens " f"across {num_topics} topics"
    )

    # ── Step 3: Load model ───────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = str(next(model.parameters()).device)
    moe_cfg = get_moe_config(model)
    num_layers = moe_cfg["num_layers"]
    num_standard_experts = moe_cfg["num_standard_experts"]
    routed_top_k = moe_cfg["routed_top_k"]

    # ── Step 4: Process documents through model ──────────────────────────────
    logger.info("\nProcessing documents through model ...")
    logger.info(
        f"  top_k={moe_cfg['top_k']} (routed={routed_top_k}, "
        f"shared={moe_cfg['num_shared_experts']})"
    )

    num_docs = len(all_docs)
    num_batches = (num_docs + args.batch_size - 1) // args.batch_size
    emb_dim = num_layers * num_standard_experts

    # Collect normalized frequencies: (num_docs, num_layers * num_standard_experts)
    all_freq = np.zeros((num_docs, emb_dim), dtype=np.float32)
    t0 = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, num_docs)
        batch_docs = all_docs[batch_start:batch_end]

        batch_freq = process_batch(
            model, batch_docs, device, num_layers, num_standard_experts, routed_top_k
        )

        all_freq[batch_start:batch_end] = batch_freq

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"  Batch {batch_idx+1}/{num_batches} "
                f"({(batch_idx+1)/num_batches:.0%})  "
                f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
            )

    # ── Step 4b: Validate frequencies ────────────────────────────────────────
    logger.info("\nValidating frequencies ...")
    assert all_freq.min() >= 0.0, f"Frequencies have negative values: min={all_freq.min()}"
    assert all_freq.max() <= 1.0, f"Frequencies exceed 1.0: max={all_freq.max()}"

    # Each layer's frequencies should sum to routed_top_k
    freq_per_layer = all_freq.reshape(num_docs, num_layers, num_standard_experts)
    layer_freq_sums = freq_per_layer.sum(axis=2)
    assert np.allclose(layer_freq_sums, routed_top_k, atol=1e-4), (
        f"Per-layer freq sums not close to {routed_top_k}: "
        f"range [{layer_freq_sums.min():.4f}, {layer_freq_sums.max():.4f}]"
    )

    logger.info(
        f"  All freq in [0, 1]: PASSED (range [{all_freq.min():.4f}, {all_freq.max():.4f}])"
    )
    logger.info(
        f"  Per-layer freq sums == {routed_top_k}: PASSED "
        f"(range [{layer_freq_sums.min():.4f}, {layer_freq_sums.max():.4f}])"
    )
    logger.info(f"  Freq shape: {all_freq.shape}")

    # ── Step 4c: Per-topic expert coverage statistics ───────────────────────
    logger.info("\nPer-topic expert coverage statistics:")
    all_labels_arr = np.array(all_labels)
    topic_stats = {}

    for topic in sorted(set(all_labels)):
        mask = all_labels_arr == topic
        # (num_docs_in_topic, num_layers, num_standard_experts)
        topic_freq = all_freq[mask].reshape(-1, num_layers, num_standard_experts)
        n_docs = topic_freq.shape[0]

        # (1) Average number of nonzero experts per layer per document
        nonzero_per_doc_layer = (topic_freq > 0).sum(axis=2)  # (n_docs, num_layers)
        avg_nonzero_per_layer = nonzero_per_doc_layer.mean(axis=0)  # (num_layers,)

        # (2) Entropy of expert distribution per layer per document, then average
        # Normalize each doc's layer freqs to a distribution (sum to 1)
        layer_sums = topic_freq.sum(axis=2, keepdims=True)  # (n_docs, num_layers, 1)
        layer_dists = topic_freq / layer_sums  # (n_docs, num_layers, num_standard_experts)
        # Entropy per doc per layer (base 2 for bits)
        entropy_per_doc_layer = np.array(
            [
                [
                    scipy_entropy(layer_dists[d, layer_idx], base=2)
                    for layer_idx in range(num_layers)
                ]
                for d in range(n_docs)
            ]
        )  # (n_docs, num_layers)
        avg_entropy_per_layer = entropy_per_doc_layer.mean(axis=0)  # (num_layers,)
        max_entropy = np.log2(num_standard_experts)

        topic_stats[topic] = {
            "num_docs": int(n_docs),
            "avg_experts_per_layer": avg_nonzero_per_layer.tolist(),
            "avg_experts_per_layer_mean": float(avg_nonzero_per_layer.mean()),
            "entropy_per_layer": avg_entropy_per_layer.tolist(),
            "entropy_per_layer_mean": float(avg_entropy_per_layer.mean()),
            "max_entropy": float(max_entropy),
        }

        logger.info(f"  [{topic}] ({n_docs} docs)")
        logger.info(
            f"    Avg experts/layer: {avg_nonzero_per_layer.mean():.1f} "
            f"(min={avg_nonzero_per_layer.min():.1f}, "
            f"max={avg_nonzero_per_layer.max():.1f}) "
            f"out of {num_standard_experts}"
        )
        logger.info(
            f"    Entropy/layer:     {avg_entropy_per_layer.mean():.2f} bits "
            f"(min={avg_entropy_per_layer.min():.2f}, "
            f"max={avg_entropy_per_layer.max():.2f}) "
            f"out of {max_entropy:.2f} max"
        )

    # Summary table sorted by mean entropy (ascending = most concentrated)
    logger.info("\n" + "=" * 90)
    logger.info(
        f"{'TOPIC':<35} {'DOCS':>5} {'AVG_EXPERTS/LAYER':>18} "
        f"{'ENTROPY (bits)':>15} {'/ MAX':>7}"
    )
    logger.info("=" * 90)
    for topic in sorted(topic_stats, key=lambda t: topic_stats[t]["entropy_per_layer_mean"]):
        ts = topic_stats[topic]
        logger.info(
            f"{topic:<35} {ts['num_docs']:>5} "
            f"{ts['avg_experts_per_layer_mean']:>18.1f} "
            f"{ts['entropy_per_layer_mean']:>15.2f} "
            f"/ {ts['max_entropy']:.2f}"
        )
    logger.info("=" * 90)

    # ── Step 5: Save results ─────────────────────────────────────────────────
    # Save normalized frequencies
    freq_path = os.path.join(args.output_dir, "expert_freq.npy")
    np.save(freq_path, all_freq)
    logger.info(f"\nSaved frequencies -> {freq_path}  shape={all_freq.shape}")

    # Save per-topic stats
    stats_path = os.path.join(args.output_dir, "topic_stats.json")
    with open(stats_path, "w") as f:
        json.dump(topic_stats, f, indent=2)
    logger.info(f"Saved topic stats -> {stats_path}")

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
        "num_shared_experts": moe_cfg["num_shared_experts"],
        "top_k": moe_cfg["top_k"],
        "routed_top_k": routed_top_k,
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
