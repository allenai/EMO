"""
Extract per-TOKEN expert-usage fingerprints from an MoE model on the
cc_all_dressed weborganizer topic mix.

This is the token-level analogue of ``extract_document.py``. Where that script
averages each document's per-token routing into a single per-document vector,
this one keeps every token as its own row. The point is to repeat the
cross-model expert-matching analysis (``match_experts.py``) on a *token*
fingerprint -- "which tokens does this expert fire on" -- instead of a
*document* fingerprint -- "which documents does this expert work on".

Output file contract is deliberately identical to ``extract_document.py`` so
that the entire downstream stack (``doc_embeddings.py`` loaders,
``match_experts.py``, ``assert_same_doc_set``) runs UNCHANGED -- the only
difference is that axis 0 is now tokens rather than documents:

    embeddings_doc_probs.npy      (num_tokens, num_layers * num_standard_experts)
                                  per token, per layer: softmax prob over
                                  standard experts (each layer slice sums to 1.0)
    embeddings_doc_topk_freq.npy  (num_tokens, num_layers * num_standard_experts)
                                  per token, per layer: 1.0 for each of the
                                  routed_top_k selected experts (each layer
                                  slice sums to routed_top_k)
    metadata_docs.jsonl.gz        one row per TOKEN: {doc_index, source,
                                  token_pos, doc_len}
    info.json                     extraction config + model dims

Cross-model validity: all models share the tokenizer and, given the same
``mix_composition.json`` + ``--shuffle-seed`` + ``--target-tokens``, load the
identical documents in the identical order. We then deterministically select
the same documents (balanced across topics) and the same first
``--tokens-per-doc`` tokens of each, so token row *t* is literally the same
token for every model. ``assert_same_doc_set`` (which compares the per-row
``source`` labels) enforces this.

Default sampling: the first ``--tokens-per-doc`` (=100) tokens of each of
``--max-tokens // tokens-per-doc`` documents, drawn evenly across the topics,
so the budget is spread over as many distinct documents as possible rather
than concentrated in a few long ones.

Usage:
    python -u -m src.scripts.clustering.extract_document_tokens \\
        --model-path models_sizescaling/<run>/step30995-hf \\
        --output-dir claude_outputs/models_sizescaling/weborganizer_tokens/<run> \\
        --composition-file claude_outputs/models_sizescaling/weborganizer/mix_composition.json \\
        --tokens-per-doc 100 --max-tokens 1000000 --batch-size 64
"""

import argparse
import gzip
import json
import logging
import math
import os
import time
from typing import Any, List, Tuple

import numpy as np
import torch

from src.scripts.clustering.extract_document import generate_uniform_composition
from src.scripts.clustering.utils import (
    EOS_TOKEN_ID,
    get_moe_config,
    load_model_and_tokenizer,
    load_source_documents_shuffled,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def process_batch_tokens(
    model: Any,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
    routed_top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a batch through the model and return PER-TOKEN routing.

    Each doc is already truncated to its first <=tokens_per_doc tokens, so all
    real (non-pad) tokens are kept. Rows are emitted in (doc, token) order:
    doc 0's tokens first, then doc 1's, etc. -- matching the metadata order
    built in main().

    Returns (probs, topk) each shaped (sum_doc_lens, num_layers * num_standard_experts).
      probs: per-token softmax over standard experts (layer slice sums to 1.0)
      topk:  per-token one-hot of the routed_top_k selected experts (sums to k)
    """
    B = len(batch_docs)
    doc_lens = [len(d) for d in batch_docs]
    max_len = max(doc_lens)

    input_ids = torch.full((B, max_len), EOS_TOKEN_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, doc in enumerate(batch_docs):
        L = len(doc)
        input_ids[i, :L] = torch.from_numpy(doc.astype(np.int64)).to(device)
        attention_mask[i, :L] = 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
        )

    # (B, max_len, num_layers, E) for both quantities.
    probs_all = torch.zeros(
        B, max_len, num_layers, num_standard_experts, dtype=torch.float32, device=device
    )
    topk_all = torch.zeros(
        B, max_len, num_layers, num_standard_experts, dtype=torch.float32, device=device
    )
    for layer_idx, layer_logits in enumerate(outputs.router_logits):
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts].float()
        probs_all[:, :, layer_idx, :] = torch.softmax(logits, dim=-1)
        top_idx = logits.topk(routed_top_k, dim=-1).indices  # (B, max_len, k)
        topk_all[:, :, layer_idx, :].scatter_(-1, top_idx, 1.0)

    probs_np = probs_all.cpu().numpy()
    topk_np = topk_all.cpu().numpy()
    del outputs, probs_all, topk_all
    torch.cuda.empty_cache()

    # Gather only the real tokens of each doc, in order.
    emb_dim = num_layers * num_standard_experts
    probs_rows = np.concatenate(
        [probs_np[i, : doc_lens[i]].reshape(doc_lens[i], emb_dim) for i in range(B)], axis=0
    )
    topk_rows = np.concatenate(
        [topk_np[i, : doc_lens[i]].reshape(doc_lens[i], emb_dim) for i in range(B)], axis=0
    )
    return probs_rows.astype(np.float32), topk_rows.astype(np.float32)


def select_balanced_docs(
    all_docs: List[np.ndarray],
    all_labels: List[str],
    topics: List[str],
    max_tokens: int,
    tokens_per_doc: int,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Pick documents evenly across topics so that, taking the first
    `tokens_per_doc` tokens of each, the total token budget is ~max_tokens but
    spread over as many distinct documents as possible.

    `all_docs` is topic-blocked (extract_document loads topic by topic), and
    within each topic the order is the seeded shuffle, so "first N per topic"
    is deterministic and identical across models.
    """
    target_docs = max(1, math.ceil(max_tokens / tokens_per_doc))
    per_topic = max(1, math.ceil(target_docs / max(1, len(topics))))

    by_topic = {t: [] for t in topics}
    for doc, label in zip(all_docs, all_labels):
        if label in by_topic:
            by_topic[label].append(doc)

    sel_docs, sel_labels = [], []
    for t in topics:
        kept = by_topic[t][:per_topic]
        for doc in kept:
            sel_docs.append(doc[:tokens_per_doc])
            sel_labels.append(t)

    logger.info(
        f"Selected {len(sel_docs)} docs ({per_topic}/topic target across "
        f"{len(topics)} topics), first {tokens_per_doc} tokens each"
    )
    return sel_docs, sel_labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-token expert-usage fingerprints on weborganizer topics"
    )
    parser.add_argument("--model-path", required=True, help="Path to HF MoE model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output dir for embeddings + metadata")
    parser.add_argument(
        "--composition-file",
        default="claude_outputs/clustering/weborganizer/mix_composition.json",
        help="Shared mix_composition.json (same docs across models). Generated if missing.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=20_000_000,
        help="Token budget for S3 doc loading (kept identical to the doc-level "
        "extraction so the same document pool is drawn).",
    )
    parser.add_argument(
        "--tokens-per-doc",
        type=int,
        default=100,
        help="Keep only the first N tokens of each selected document.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1_000_000,
        help="Approx total token-rows to emit (spread over max-tokens/tokens-per-doc docs).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-doc-len", type=int, default=32)
    parser.add_argument("--max-doc-len", type=int, default=2048)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Use only the first 2 topics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: composition ──────────────────────────────────────────────────
    if os.path.exists(args.composition_file):
        logger.info(f"Loading composition from {args.composition_file}")
        with open(args.composition_file) as f:
            composition = json.load(f)
    else:
        logger.info(f"No composition at {args.composition_file}; generating ...")
        composition = generate_uniform_composition(args.composition_file)

    sources = composition["sources"]
    if args.debug:
        sources = dict(list(sorted(sources.items()))[:2])
        logger.info(f"DEBUG: using 2 topics: {list(sources.keys())}")
    num_topics = len(sources)
    tokens_per_topic = args.target_tokens // num_topics

    # ── Step 2: load docs (identical pool to the doc-level extraction) ────────
    logger.info("\nLoading documents from S3 (shuffled, seed=%d) ..." % args.shuffle_seed)
    all_docs: List[np.ndarray] = []
    all_labels: List[str] = []
    for topic, info in sorted(sources.items()):
        docs = load_source_documents_shuffled(
            info["all_files"],
            tokens_per_topic,
            min_doc_len=args.min_doc_len,
            max_doc_len=args.max_doc_len,
            seed=args.shuffle_seed,
        )
        logger.info(f"  [{topic}] loaded {len(docs)} docs")
        all_docs.extend(docs)
        all_labels.extend([topic] * len(docs))

    topics = sorted(sources.keys())
    sel_docs, sel_labels = select_balanced_docs(
        all_docs, all_labels, topics, args.max_tokens, args.tokens_per_doc
    )
    doc_lens = [len(d) for d in sel_docs]
    total_tokens = int(sum(doc_lens))
    logger.info(f"Total token-rows to emit: {total_tokens:,} across {len(sel_docs)} docs")

    # ── Step 3: load model ───────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if hasattr(model, "hf_device_map"):
        first_device = next(iter(model.hf_device_map.values()))
        device = f"cuda:{first_device}" if isinstance(first_device, int) else str(first_device)
    else:
        device = str(next(model.parameters()).device)
    logger.info(f"Input device: {device}")

    moe_cfg = get_moe_config(model)
    num_layers = moe_cfg["num_layers"]
    num_standard_experts = moe_cfg["num_standard_experts"]
    routed_top_k = moe_cfg["routed_top_k"]
    emb_dim = num_layers * num_standard_experts

    # ── Step 4: forward pass ─────────────────────────────────────────────────
    num_docs = len(sel_docs)
    num_batches = (num_docs + args.batch_size - 1) // args.batch_size
    probs_arr = np.zeros((total_tokens, emb_dim), dtype=np.float32)
    topk_arr = np.zeros((total_tokens, emb_dim), dtype=np.float32)

    logger.info(
        f"\nProcessing {num_docs} docs ({total_tokens:,} tokens) in {num_batches} batches "
        f"(routed_top_k={routed_top_k}, experts={num_standard_experts}, layers={num_layers})"
    )
    t0 = time.time()
    row = 0
    for batch_idx in range(num_batches):
        bs = batch_idx * args.batch_size
        be = min(bs + args.batch_size, num_docs)
        batch = sel_docs[bs:be]
        pr, tk = process_batch_tokens(
            model, batch, device, num_layers, num_standard_experts, routed_top_k
        )
        n = pr.shape[0]
        probs_arr[row : row + n] = pr
        topk_arr[row : row + n] = tk
        row += n

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"  Batch {batch_idx+1}/{num_batches} ({(batch_idx+1)/num_batches:.0%})  "
                f"rows={row:,}/{total_tokens:,}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
            )
    assert row == total_tokens, f"row count {row} != expected {total_tokens}"

    # ── Step 5: validate ─────────────────────────────────────────────────────
    logger.info("\nValidating embeddings ...")
    pr_layer_sums = probs_arr.reshape(total_tokens, num_layers, num_standard_experts).sum(axis=2)
    tk_layer_sums = topk_arr.reshape(total_tokens, num_layers, num_standard_experts).sum(axis=2)
    assert probs_arr.min() >= 0.0, f"probs negative: {probs_arr.min()}"
    assert np.allclose(pr_layer_sums, 1.0, atol=1e-3), (
        f"probs layer sums not ~1.0: [{pr_layer_sums.min():.4f}, {pr_layer_sums.max():.4f}]"
    )
    assert np.allclose(tk_layer_sums, routed_top_k, atol=1e-3), (
        f"topk layer sums not ~{routed_top_k}: [{tk_layer_sums.min():.4f}, {tk_layer_sums.max():.4f}]"
    )
    logger.info(
        f"  probs: range=[{probs_arr.min():.6f}, {probs_arr.max():.4f}]  "
        f"topk layer-sum range=[{tk_layer_sums.min():.2f}, {tk_layer_sums.max():.2f}]"
    )

    # ── Step 6: save (same file contract as extract_document.py) ──────────────
    pr_path = os.path.join(args.output_dir, "embeddings_doc_probs.npy")
    tk_path = os.path.join(args.output_dir, "embeddings_doc_topk_freq.npy")
    np.save(pr_path, probs_arr)
    np.save(tk_path, topk_arr)
    logger.info(f"\nSaved {pr_path}  shape={probs_arr.shape}")
    logger.info(f"Saved {tk_path}  shape={topk_arr.shape}")

    meta_path = os.path.join(args.output_dir, "metadata_docs.jsonl.gz")
    with gzip.open(meta_path, "wt") as f:
        for doc_index, (label, dlen) in enumerate(zip(sel_labels, doc_lens)):
            for pos in range(dlen):
                f.write(
                    json.dumps(
                        {
                            "doc_index": doc_index,
                            "source": label,
                            "token_pos": pos,
                            "doc_len": int(dlen),
                        }
                    )
                    + "\n"
                )
    logger.info(f"Saved {meta_path}")

    info_out = {
        "model_path": args.model_path,
        "level": "token",
        "num_layers": num_layers,
        "num_standard_experts": num_standard_experts,
        "num_shared_experts": moe_cfg["num_shared_experts"],
        "top_k": moe_cfg["top_k"],
        "routed_top_k": routed_top_k,
        "num_topics": num_topics,
        "tokens_per_doc": args.tokens_per_doc,
        "max_tokens": args.max_tokens,
        "total_docs": num_docs,
        "total_tokens": total_tokens,
        "shuffle_seed": args.shuffle_seed,
        "topics": topics,
        "composition_file": args.composition_file,
    }
    with open(os.path.join(args.output_dir, "info.json"), "w") as f:
        json.dump(info_out, f, indent=2)
    logger.info("Saved info.json\nDone.")


if __name__ == "__main__":
    main()
