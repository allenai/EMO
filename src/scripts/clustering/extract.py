"""
Extract per-token router logits from a trained MoE model.

Saves raw router logits — the single primitive from which all other
representations (probs, topk, doc-level aggregation) are derived
in transform.py.

Supports three data sources:
  - pretraining: S3-based shuffled sampling with per-document truncation
  - mmlu: all 57 MMLU subjects (validation split by default)
  - hellaswag: all HellaSwag splits (train/validation/test)

Output format is identical across sources:
  embeddings_logits.npy      (num_tokens, num_layers * num_experts), float16
  documents.npy              flat token IDs (int32)
  doc_boundaries.npy         (num_docs + 1,) cumulative offsets
  metadata_tokens.jsonl.gz   per-token metadata
  metadata_docs.jsonl.gz     per-document metadata
  info.json                  extraction config

Usage:
    # Pretraining (shuffled, truncated)
    python -m src.scripts.clustering.extract \\
        --source pretraining \\
        --model-path models/.../step30995-hf \\
        --composition-file .../pretraining_mix.json \\
        --output-dir .../pretraining/<model>/ \\
        --target-tokens 1000000 --max-tokens-per-doc 100

    # MMLU
    python -m src.scripts.clustering.extract \\
        --source mmlu --model-path models/.../step30995-hf \\
        --output-dir .../mmlu/<model>/

    # HellaSwag
    python -m src.scripts.clustering.extract \\
        --source hellaswag --model-path models/.../step30995-hf \\
        --output-dir .../hellaswag/<model>/
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
import torch

from src.scripts.clustering.utils import (
    EOS_TOKEN_ID,
    get_moe_config,
    load_model_and_tokenizer,
    load_source_documents_shuffled,
    tokenize_prompts,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_logits(
    model,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
) -> tuple:
    """
    Run a batch through the model and extract per-token router logits.

    Returns:
        logits: (T_valid, num_layers * num_standard_experts) float16 array
        token_info: list of (batch_doc_idx, token_position) per valid token
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

    # Collect per-layer logits and extract valid tokens
    valid = attention_mask.bool()  # (B, S)
    layers = []
    for layer_logits in outputs.router_logits:
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts]
        layers.append(logits[valid].float())  # (T_valid, E)
    all_logits = torch.cat(layers, dim=-1).cpu().numpy().astype(np.float16)

    # Build token position info
    token_info = []
    for i in range(B):
        positions = valid[i].nonzero(as_tuple=True)[0].cpu().tolist()
        for pos in positions:
            token_info.append((i, pos))

    torch.cuda.empty_cache()
    return all_logits, token_info


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(model, tokenizer, sample_docs, device,
                      num_layers, num_standard_experts, routed_top_k):
    """Quick sanity check on a single document."""
    logger.info("=== Running sanity checks ===")

    doc = sample_docs[0]
    D = num_layers * num_standard_experts
    doc_len = len(doc)

    logits, token_info = extract_logits(
        model, [doc], device, num_layers, num_standard_experts,
    )

    # Shape check
    assert logits.shape == (doc_len, D), \
        f"Logits shape {logits.shape} != ({doc_len}, {D})"
    assert logits.dtype == np.float16
    assert not np.isnan(logits).any(), "Logits contain NaN"
    logger.info(f"  Shape: ({doc_len}, {D}), range=[{logits.min():.4f}, {logits.max():.4f}]")

    # Token info length
    assert len(token_info) == doc_len, \
        f"token_info length {len(token_info)} != {doc_len}"

    # Decode sample
    decoded = tokenizer.decode(doc[:80].tolist(), skip_special_tokens=True)
    logger.info(f"  Sample text: {decoded[:120]!r}")

    logger.info("=== Sanity checks passed ===\n")


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_pretraining_docs(args) -> List[tuple]:
    """
    Load documents from pretraining data via S3.

    Returns list of (source_label, docs_list) pairs.
    """
    with open(args.composition_file) as f:
        composition = json.load(f)

    sources = composition["sources"]
    logger.info(f"Loaded composition for {len(sources)} sources")

    # Compute effective target: if truncating, oversample to compensate
    effective_target = args.target_tokens
    if args.max_tokens_per_doc > 0:
        effective_target = args.target_tokens * 8  # oversample for truncation
        logger.info(f"Oversampling: {args.target_tokens} post-truncation -> "
                     f"{effective_target} pre-truncation (8x)")

    result = []
    for label, info in sorted(sources.items(), key=lambda x: -x[1]["fraction"]):
        alloc = int(info["fraction"] * effective_target)
        if alloc < 32:
            continue
        logger.info(f"  {label}: {info['fraction']:.2%} -> {alloc:,} tokens")

        docs = load_source_documents_shuffled(
            info["all_files"], alloc,
            min_doc_len=32, max_doc_len=2048,
            seed=args.shuffle_seed,
        )

        if args.max_tokens_per_doc > 0:
            docs = [doc[:args.max_tokens_per_doc] for doc in docs]

        if docs:
            result.append((label, docs))

    return result


def load_mmlu_docs(args) -> List[tuple]:
    """Load MMLU prompts as documents. Returns list of (subject, docs) pairs."""
    from src.hf_training.data_utils import get_formatted_prompts
    from src.offline_evals.tasks.splits_mmlu import MMLU_CATEGORIES

    # Build subject list
    all_subjects = []
    subject_to_category = {}
    for cat, subjects in MMLU_CATEGORIES.items():
        for s in subjects:
            subject_to_category[s] = cat
            all_subjects.append(s)
    all_subjects.sort()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects = all_subjects

    logger.info(f"Loading {len(subjects)} MMLU subjects ({args.mmlu_split} split)")

    # We need the tokenizer for tokenization — load lazily via args
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = args.max_tokens_per_doc if args.max_tokens_per_doc > 0 else 2048

    result = []
    for subject in subjects:
        prompts, _ = get_formatted_prompts(f"mmlu_{subject}", args.mmlu_split)
        docs = tokenize_prompts(prompts, tokenizer, max_length=max_len)
        if docs:
            # Attach category as extra metadata
            result.append((subject, docs))
            logger.info(f"  {subject}: {len(docs)} prompts")

    # Store subject_to_category for later metadata saving
    args._subject_to_category = subject_to_category
    return result


def load_hellaswag_docs(args) -> List[tuple]:
    """Load HellaSwag prompts as documents. Returns list of (split, docs) pairs."""
    from src.hf_training.data_utils import get_formatted_prompts

    splits = [s.strip() for s in args.hellaswag_splits.split(",")]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = args.max_tokens_per_doc if args.max_tokens_per_doc > 0 else 2048

    result = []
    for split in splits:
        prompts, _ = get_formatted_prompts("hellaswag", split)
        docs = tokenize_prompts(prompts, tokenizer, max_length=max_len)
        if docs:
            result.append((split, docs))
            logger.info(f"  {split}: {len(docs)} prompts")

    return result


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(source_docs, model, tokenizer, device, args,
                   num_layers, num_standard_experts, routed_top_k, emb_dim):
    """
    Extract token-level logits from all source documents and save outputs.

    Args:
        source_docs: list of (source_label, docs_list) pairs
    """
    all_logits: List[np.ndarray] = []
    all_meta: List[dict] = []
    all_doc_tokens: List[np.ndarray] = []
    all_doc_sources: List[str] = []
    all_doc_extra: List[dict] = []  # source-specific extra fields
    global_doc_idx = 0

    subject_to_category = getattr(args, '_subject_to_category', {})

    for label, docs in source_docs:
        logger.info(f"\nSource: {label}  docs={len(docs)}  "
                     f"tokens={sum(len(d) for d in docs):,}")

        for doc in docs:
            all_doc_tokens.append(doc)
            all_doc_sources.append(label)
            extra = {}
            if args.source == "mmlu" and label in subject_to_category:
                extra["category"] = subject_to_category[label]
            all_doc_extra.append(extra)

        num_batches = (len(docs) + args.batch_size - 1) // args.batch_size
        source_tokens = 0
        t0 = time.time()

        for batch_idx, i in enumerate(range(0, len(docs), args.batch_size)):
            batch = docs[i : i + args.batch_size]
            logits, token_info = extract_logits(
                model, batch, device, num_layers, num_standard_experts,
            )
            all_logits.append(logits)

            for batch_doc_idx, token_pos in token_info:
                doc = batch[batch_doc_idx]
                all_meta.append({
                    "source": label,
                    "doc_index": global_doc_idx + i + batch_doc_idx,
                    "token_position": token_pos,
                    "token_id": int(doc[token_pos]),
                })
            source_tokens += len(token_info)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{label}] batch {batch_idx+1}/{num_batches} "
                    f"({(batch_idx+1)/num_batches:.0%})  "
                    f"tokens={source_tokens:,}  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
                )

        global_doc_idx += len(docs)
        logger.info(f"  Done: {source_tokens:,} tokens from {len(docs)} docs")

    # --- Save outputs ---

    # Logits
    logits_arr = np.concatenate(all_logits, axis=0)
    out_logits = os.path.join(args.output_dir, "embeddings_logits.npy")
    np.save(out_logits, logits_arr)
    logger.info(f"Logits: {out_logits}  shape={logits_arr.shape}  dtype={logits_arr.dtype}")

    # Documents + boundaries (for context recovery and aggregation)
    boundaries = np.zeros(len(all_doc_tokens) + 1, dtype=np.int64)
    for i, doc in enumerate(all_doc_tokens):
        boundaries[i + 1] = boundaries[i] + len(doc)
    flat_tokens = np.concatenate(all_doc_tokens).astype(np.int32)

    np.save(os.path.join(args.output_dir, "documents.npy"), flat_tokens)
    np.save(os.path.join(args.output_dir, "doc_boundaries.npy"), boundaries)
    logger.info(f"Documents: {flat_tokens.shape[0]:,} tokens, {len(all_doc_tokens)} docs")

    # Per-token metadata
    out_meta = os.path.join(args.output_dir, "metadata_tokens.jsonl.gz")
    with gzip.open(out_meta, "wt") as f:
        for m in all_meta:
            f.write(json.dumps(m) + "\n")

    # Per-document metadata
    out_doc_meta = os.path.join(args.output_dir, "metadata_docs.jsonl.gz")
    with gzip.open(out_doc_meta, "wt") as f:
        for i, source in enumerate(all_doc_sources):
            doc_len = int(boundaries[i + 1] - boundaries[i])
            entry = {"doc_index": i, "source": source, "doc_len": doc_len}
            entry.update(all_doc_extra[i])
            f.write(json.dumps(entry) + "\n")

    # Info
    source_token_counts = defaultdict(int)
    for m in all_meta:
        source_token_counts[m["source"]] += 1
    source_doc_counts = defaultdict(int)
    for s in all_doc_sources:
        source_doc_counts[s] += 1

    info_out = {
        "source": args.source,
        "num_tokens": len(all_meta),
        "num_docs": global_doc_idx,
        "num_layers": num_layers,
        "num_standard_experts": num_standard_experts,
        "routed_top_k": routed_top_k,
        "emb_dim": emb_dim,
        "model_path": args.model_path,
        "source_token_counts": dict(source_token_counts),
        "source_doc_counts": dict(source_doc_counts),
        "max_tokens_per_doc": args.max_tokens_per_doc,
    }
    if args.source == "pretraining":
        info_out["target_tokens"] = args.target_tokens
        info_out["shuffle_seed"] = args.shuffle_seed
    if args.source == "mmlu":
        info_out["mmlu_split"] = args.mmlu_split
    if args.source == "hellaswag":
        info_out["hellaswag_splits"] = args.hellaswag_splits

    out_info = os.path.join(args.output_dir, "info.json")
    with open(out_info, "w") as f:
        json.dump(info_out, f, indent=2)

    logger.info(f"\nSaved {len(all_meta):,} token embeddings from {global_doc_idx} docs")
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-token router logits from a trained MoE model"
    )
    parser.add_argument("--source", required=True,
                        choices=["pretraining", "mmlu", "hellaswag"],
                        help="Data source to extract from")
    parser.add_argument("--model-path", required=True,
                        help="Path to HF model checkpoint")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens-per-doc", type=int, default=0,
                        help="Truncate each document to this many tokens. "
                             "0 = no truncation (default: 0)")

    # Pretraining-specific
    parser.add_argument("--composition-file", default=None,
                        help="mix_composition.json (required for --source pretraining)")
    parser.add_argument("--target-tokens", type=int, default=1_000_000,
                        help="Target token count for pretraining (default: 1M)")
    parser.add_argument("--shuffle-seed", type=int, default=42)

    # MMLU-specific
    parser.add_argument("--mmlu-split", default="validation",
                        choices=["validation", "test", "train"])
    parser.add_argument("--subjects", default=None,
                        help="Comma-separated MMLU subjects (default: all 57)")

    # HellaSwag-specific
    parser.add_argument("--hellaswag-splits", default="train,validation,test",
                        help="Comma-separated splits (default: train,validation,test)")

    args = parser.parse_args()

    if args.source == "pretraining" and args.composition_file is None:
        parser.error("--composition-file is required for --source pretraining")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = str(next(model.parameters()).device)

    moe_cfg = get_moe_config(model)
    num_layers = moe_cfg["num_layers"]
    num_standard_experts = moe_cfg["num_standard_experts"]
    routed_top_k = moe_cfg["routed_top_k"]
    emb_dim = moe_cfg["emb_dim"]

    # Load source documents
    logger.info(f"\nLoading documents from source: {args.source}")
    if args.source == "pretraining":
        source_docs = load_pretraining_docs(args)
    elif args.source == "mmlu":
        source_docs = load_mmlu_docs(args)
    elif args.source == "hellaswag":
        source_docs = load_hellaswag_docs(args)

    if not source_docs:
        logger.error("No documents loaded, exiting.")
        return

    # Collect sample docs for sanity check
    sample_docs = [source_docs[0][1][0]]
    run_sanity_checks(model, tokenizer, sample_docs, device,
                      num_layers, num_standard_experts, routed_top_k)

    # Run extraction
    run_extraction(source_docs, model, tokenizer, device, args,
                   num_layers, num_standard_experts, routed_top_k, emb_dim)


if __name__ == "__main__":
    main()
