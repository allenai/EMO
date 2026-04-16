"""
Extract router embeddings from HellaSwag data (all splits).

Loads train, validation, and test splits via get_formatted_prompts(),
tokenizes them, runs through the MoE model, and saves embeddings in the
same format as extract_router_embeddings.py so that transform_and_cluster.py
works unchanged.

The "source" field in metadata is the split name ("train", "validation", "test"),
so downstream code can identify which embedding rows belong to which split.

Usage:
    python -u -m src.scripts.analysis.extract_router_embeddings_hellaswag \\
        --model-path models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf \\
        --output-dir claude_outputs/analysis/router_clustering_hellaswag/MODEL_NAME \\
        --batch-size 32
"""

import argparse
import gzip
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.hf_training.data_utils import get_formatted_prompts
from src.scripts.analysis.extract_router_embeddings import (
    EMBEDDING_REGISTRY,
    embed_batch,
)
from src.scripts.analysis.utils import get_moe_config, load_model_and_tokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_SPLITS = ["train", "validation", "test"]


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


def main():
    parser = argparse.ArgumentParser(description="Extract router embeddings from HellaSwag data")
    parser.add_argument("--model-path", required=True, help="Path to HF model checkpoint")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for embeddings and metadata"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-doc-len", type=int, default=2048, help="Max token length per prompt (default: 2048)"
    )
    parser.add_argument(
        "--embeddings",
        default="all",
        help="Comma-separated embedding types or 'all' (default: all)",
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated splits to process (default: train,validation,test)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve embedding types
    all_names = sorted(EMBEDDING_REGISTRY.keys())
    if args.embeddings == "all":
        requested_names = all_names
    else:
        requested_names = [s.strip() for s in args.embeddings.split(",")]
        for name in requested_names:
            if name not in EMBEDDING_REGISTRY:
                parser.error(f"Unknown embedding type '{name}'. Available: {', '.join(all_names)}")
    embedding_types = [EMBEDDING_REGISTRY[name] for name in requested_names]
    logger.info(f"Will compute embeddings: {[et.name for et in embedding_types]}")

    # Resolve splits
    splits = [s.strip() for s in args.splits.split(",")]
    for s in splits:
        if s not in ALL_SPLITS:
            parser.error(f"Unknown split '{s}'. Available: {', '.join(ALL_SPLITS)}")
    logger.info(f"Will process splits: {splits}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = str(next(model.parameters()).device)

    moe_cfg = get_moe_config(model)
    num_layers = moe_cfg["num_layers"]
    num_standard_experts = moe_cfg["num_standard_experts"]
    routed_top_k = moe_cfg["routed_top_k"]
    emb_dim = moe_cfg["emb_dim"]

    logger.info(f"Model: {num_layers} layers, {num_standard_experts} standard experts")
    logger.info(f"Embedding dim: {emb_dim}, routed top-k: {routed_top_k}")

    # Extract embeddings per split
    all_embeddings: Dict[str, List[np.ndarray]] = {et.name: [] for et in embedding_types}
    all_meta: List[dict] = []
    total_tokens = 0

    for split in splits:
        logger.info(f"\nLoading hellaswag {split} ...")
        prompts, req_type = get_formatted_prompts("hellaswag", split)
        docs = tokenize_prompts(prompts, tokenizer, args.max_doc_len)

        if not docs:
            logger.warning(f"  No documents for {split}, skipping")
            continue

        split_tokens = sum(len(d) for d in docs)
        total_tokens += split_tokens
        logger.info(f"  {len(docs)} prompts, {split_tokens:,} tokens (request_type={req_type})")

        # Batch inference
        batch_embeddings: Dict[str, List[np.ndarray]] = {et.name: [] for et in embedding_types}
        num_batches = (len(docs) + args.batch_size - 1) // args.batch_size
        t0 = time.time()

        for batch_idx, i in enumerate(range(0, len(docs), args.batch_size)):
            batch = docs[i : i + args.batch_size]
            results = embed_batch(
                model,
                batch,
                device,
                num_layers,
                num_standard_experts,
                embedding_types,
                routed_top_k=routed_top_k,
            )
            for name, arr in results.items():
                batch_embeddings[name].append(arr)

            if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{split}] batch {batch_idx+1}/{num_batches} "
                    f"({(batch_idx+1)/num_batches:.0%})  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
                )

        elapsed = time.time() - t0
        logger.info(f"  Embedded {split} in {elapsed/60:.1f}m ({num_batches} batches)")

        for name in batch_embeddings:
            all_embeddings[name].append(np.concatenate(batch_embeddings[name], axis=0))

        # Track metadata: source is the split name
        for doc, prompt in zip(docs, prompts):
            all_meta.append(
                {
                    "source": split,
                    "doc_len": int(len(doc)),
                    "preview": prompt[:3000],
                }
            )

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
    source_counts: dict[str, int] = defaultdict(int)
    for m in all_meta:
        source_counts[m["source"]] += 1

    info_out = {
        "granularity": "document",
        "dataset": "hellaswag",
        "splits": splits,
        "num_docs": len(all_meta),
        "total_tokens": total_tokens,
        "num_layers": num_layers,
        "num_standard_experts": num_standard_experts,
        "routed_top_k": routed_top_k,
        "emb_dim": emb_dim,
        "model_path": args.model_path,
        "embedding_types": [et.name for et in embedding_types],
        "source_doc_counts": dict(source_counts),
    }
    out_info = os.path.join(args.output_dir, "info.json")
    with open(out_info, "w") as f:
        json.dump(info_out, f, indent=2)

    logger.info(f"\nSaved {len(all_meta)} document embeddings ({total_tokens:,} tokens)")
    logger.info(f"  embeddings: {args.output_dir}/embeddings_*.npy")
    logger.info(f"  metadata:   {out_meta}")
    logger.info(f"  info:       {out_info}")

    # Print summary
    logger.info("\nPer-split summary:")
    for split in splits:
        count = source_counts.get(split, 0)
        logger.info(f"  {split:<15} {count:>6} docs")


if __name__ == "__main__":
    main()
