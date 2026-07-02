"""
Extract per-document router embeddings for a pre-tokenized training-data window.

Consumes the doc JSONL.gz files produced by
scripts/modular_extension/extract_training_doc_window.py (one JSON per document with
`token_ids`, `source_path`, `doc_start_offset`, `doc_len`) and computes, for every
document, the same in-forward pooled embeddings as extract_document.py:

  - doc_probs     -- mean softmax probability per (layer, standard expert) over the
                     doc's tokens (each layer slice sums to 1.0)
  - doc_topk_freq -- normalized top-k selection frequency (each layer slice sums to
                     routed_top_k)

Documents are truncated to --max-tokens-per-doc (default 2048) for the embedding
forward pass only. No per-token data is persisted.

Sharding: documents are enumerated globally (input files sorted by name, docs in file
order); shard i takes docs i::num_shards. Shard outputs are independent and idempotent
(a shard is skipped when its outputs already exist), so a sweep can be resumed or
extended by relaunching with more shards.

Usage (one shard):
    PYTHONPATH=.:src python -u -m src.scripts.clustering.extract_doc_window \\
        --docs-glob 'modular_extension/data/<run>_100B-110B/docs-*.jsonl.gz' \\
        --model-path models_v2/<run>/step23842-hf \\
        --output-dir modular_extension/cluster/emo100b_step23842/embeddings \\
        --shards 0 --num-shards 128
"""

import argparse
import glob
import gzip
import json
import logging
import os
import time
from typing import List

import numpy as np
import torch

from .utils import EOS_TOKEN_ID, get_moe_config, load_model_and_tokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def process_batch(
    model,
    batch_docs: List[np.ndarray],
    device: str,
    num_layers: int,
    num_standard_experts: int,
    routed_top_k: int,
):
    """Same math as extract_document.process_batch (doc_probs + doc_topk_freq)."""
    B = len(batch_docs)
    max_len = max(len(d) for d in batch_docs)

    input_ids = torch.full((B, max_len), EOS_TOKEN_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, doc in enumerate(batch_docs):
        L = len(doc)
        input_ids[i, :L] = torch.from_numpy(doc.astype(np.int64)).to(device)
        attention_mask[i, :L] = 1

    # Use the base transformer (model.model), not the CausalLM wrapper: the wrapper
    # computes full-vocab logits for every position (~26GB at 128K batch tokens) and the
    # aux load-balancing loss (multi-GB one-hot) whenever output_router_logits=True.
    # The base model returns the same router_logits without either.
    base = getattr(model, "model", model)
    outputs = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_router_logits=True,
    )

    counts = torch.zeros(B, num_layers, num_standard_experts, dtype=torch.float32, device=device)
    prob_sums = torch.zeros(B, num_layers, num_standard_experts, dtype=torch.float32, device=device)
    valid_mask = attention_mask.to(torch.float32)

    for layer_idx, layer_logits in enumerate(outputs.router_logits):
        logits = layer_logits.view(B, max_len, -1)[:, :, :num_standard_experts].float()

        top_indices = logits.topk(routed_top_k, dim=-1).indices
        flat_indices = top_indices.reshape(B, -1)
        valid_repeated = (
            attention_mask.unsqueeze(-1)
            .expand(-1, -1, routed_top_k)
            .reshape(B, -1)
            .to(torch.float32)
        )
        counts[:, layer_idx, :].scatter_add_(1, flat_indices, valid_repeated)

        probs = torch.softmax(logits, dim=-1)
        masked_probs = probs * valid_mask.unsqueeze(-1)
        prob_sums[:, layer_idx, :] = masked_probs.sum(dim=1)

    doc_lens = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0).unsqueeze(-1)
    topk_freq = (counts / doc_lens).view(B, -1).cpu().numpy().astype(np.float16)
    mean_probs = (prob_sums / doc_lens).view(B, -1).cpu().numpy().astype(np.float16)
    return mean_probs, topk_freq


def iter_shard_docs(files: List[str], shard: int, num_shards: int, limit=None):
    """Yield (global_doc_index, file_index, record_dict) for docs in this shard.
    Only shard-selected lines are JSON-parsed."""
    g = 0
    yielded = 0
    for fi, path in enumerate(files):
        with gzip.open(path, "rt") as f:
            for line in f:
                if g % num_shards == shard:
                    yield g, fi, json.loads(line)
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                g += 1


def batches_by_token_budget(chunk, batch_tokens: int):
    """chunk: list of (g, fi, rec, tokens). Sort by length; yield batches with
    B * max_len <= batch_tokens (padding-aware budget)."""
    chunk = sorted(chunk, key=lambda x: len(x[3]))
    batch = []
    for item in chunk:
        cand_max = len(item[3])  # sorted ascending -> candidate max is the new item
        if batch and (len(batch) + 1) * cand_max > batch_tokens:
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def run_shard(args, model, tokenizer, moe, device, shard: int):
    out_probs = os.path.join(args.output_dir, f"doc_probs-{shard:03d}.npy")
    out_topk = os.path.join(args.output_dir, f"doc_topk_freq-{shard:03d}.npy")
    out_ids = os.path.join(args.output_dir, f"doc_ids-{shard:03d}.npz")
    out_info = os.path.join(args.output_dir, f"info-{shard:03d}.json")
    if all(os.path.exists(p) for p in (out_probs, out_topk, out_ids, out_info)):
        logger.info(f"[shard {shard}] outputs exist -- skipping")
        return

    files = sorted(glob.glob(args.docs_glob))
    assert files, f"No files match {args.docs_glob}"

    probs_rows, topk_rows = [], []
    ids_g, ids_fi, ids_off, ids_len, ids_embtok = [], [], [], [], []
    n_docs = 0
    n_tokens = 0
    t0 = time.monotonic()
    CHUNK_DOCS = 8192

    def flush(chunk):
        nonlocal n_docs, n_tokens
        for batch in batches_by_token_budget(chunk, args.batch_tokens):
            docs = [b[3] for b in batch]
            mean_probs, topk_freq = process_batch(
                model, docs, device,
                moe["num_layers"], moe["num_standard_experts"], moe["routed_top_k"],
            )
            probs_rows.append(mean_probs)
            topk_rows.append(topk_freq)
            for (g, fi, rec, toks) in batch:
                ids_g.append(g)
                ids_fi.append(fi)
                ids_off.append(rec["doc_start_offset"])
                ids_len.append(rec["doc_len"])
                ids_embtok.append(len(toks))
                n_tokens += len(toks)
            n_docs += len(batch)
        elapsed = time.monotonic() - t0
        logger.info(
            f"[shard {shard}] {n_docs:,} docs, {n_tokens:,} tokens "
            f"({n_tokens/elapsed:,.0f} tok/s, {n_docs/elapsed:,.1f} docs/s)"
        )

    # Background reader thread: gzip+JSON iteration costs ~half a forward's time, so
    # overlap it with GPU work via a small chunk queue.
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=2)

    def reader():
        chunk = []
        try:
            for g, fi, rec in iter_shard_docs(files, shard, args.num_shards, args.limit):
                toks = np.asarray(rec["token_ids"][: args.max_tokens_per_doc], dtype=np.int64)
                chunk.append((g, fi, {"doc_start_offset": rec["doc_start_offset"],
                                      "doc_len": rec["doc_len"]}, toks))
                if len(chunk) >= CHUNK_DOCS:
                    q.put(chunk)
                    chunk = []
            if chunk:
                q.put(chunk)
        finally:
            q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while True:
        chunk = q.get()
        if chunk is None:
            break
        flush(chunk)
    t.join()

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(out_probs, np.concatenate(probs_rows) if probs_rows else np.zeros((0, moe["emb_dim"]), np.float16))
    np.save(out_topk, np.concatenate(topk_rows) if topk_rows else np.zeros((0, moe["emb_dim"]), np.float16))
    np.savez(
        out_ids,
        global_doc_index=np.asarray(ids_g, dtype=np.int64),
        file_index=np.asarray(ids_fi, dtype=np.int32),
        doc_start_offset=np.asarray(ids_off, dtype=np.int64),
        doc_len=np.asarray(ids_len, dtype=np.int64),
        n_embed_tokens=np.asarray(ids_embtok, dtype=np.int32),
    )
    elapsed = time.monotonic() - t0
    with open(out_info, "w") as f:
        json.dump(
            {
                "shard": shard,
                "num_shards": args.num_shards,
                "model_path": args.model_path,
                "docs_glob": args.docs_glob,
                "files": files,
                "max_tokens_per_doc": args.max_tokens_per_doc,
                "limit": args.limit,
                "num_docs": n_docs,
                "num_embed_tokens": n_tokens,
                "elapsed_seconds": round(elapsed, 1),
                "tokens_per_second": round(n_tokens / max(elapsed, 1e-9), 1),
                **moe,
            },
            f,
            indent=2,
        )
    logger.info(f"[shard {shard}] DONE: {n_docs:,} docs, {n_tokens:,} tokens in {elapsed/60:.1f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs-glob", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shards", required=True,
                   help="comma-separated shard indices this process computes (e.g. '0' or '0,8,16')")
    p.add_argument("--num-shards", type=int, default=128)
    p.add_argument("--max-tokens-per-doc", type=int, default=2048)
    p.add_argument("--batch-tokens", type=int, default=131072,
                   help="padding-aware token budget per forward (B * max_len)")
    p.add_argument("--limit", type=int, default=None, help="docs per shard (smoke tests)")
    args = p.parse_args()

    shard_list = [int(s) for s in args.shards.split(",")]
    for s in shard_list:
        assert 0 <= s < args.num_shards

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    moe = get_moe_config(model)
    device = next(model.parameters()).device

    for s in shard_list:
        run_shard(args, model, tokenizer, moe, device, s)


if __name__ == "__main__":
    main()
