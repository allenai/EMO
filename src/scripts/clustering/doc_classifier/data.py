"""Shared data access for Stage-2 featurizers: join document token IDs to embedding rows.

Documents are stored as token IDs (dolma2-tokenizer) in `docs-*.jsonl.gz`; the embedding
row order lives in `doc_ids.npz`. A document's canonical identity is its embedding row index
`i` (aligned with the split, labels, and router embeddings). We join token IDs to `i` via the
`(source_path, doc_start_offset)` key that `export_doc_partition.py` uses.
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def build_row_map(data_dir: str) -> tuple[dict, int]:
    """Return ``{(source_path, doc_start_offset): row_i}`` and N from doc_ids.npz."""
    ids = np.load(os.path.join(data_dir, "doc_ids.npz"), allow_pickle=True)
    source_paths = [str(x) for x in ids["source_paths"]]
    si = ids["source_index"]
    off = ids["doc_start_offset"]
    n = len(si)
    row_map = {}
    for i in range(n):
        row_map[(source_paths[int(si[i])], int(off[i]))] = i
    logger.info(f"row map: {n:,} rows, {len(source_paths)} source paths")
    return row_map, n


def stream_token_ids(docs_dir: str, row_map: dict, token_cap: int, batch_size: int = 20000):
    """Yield ``(row_ids, token_id_lists)`` batches from docs-*.jsonl.gz (row_map hits only)."""
    files = sorted(glob.glob(os.path.join(docs_dir, "docs-*.jsonl.gz")))
    if not files:
        raise FileNotFoundError(f"no docs-*.jsonl.gz in {docs_dir}")
    buf_ids: list[int] = []
    buf_tok: list[list[int]] = []
    for fp in files:
        with gzip.open(fp, "rt") as f:
            for line in f:
                r = json.loads(line)
                i = row_map.get((r["source_path"], int(r["doc_start_offset"])))
                if i is None:
                    continue
                buf_ids.append(i)
                buf_tok.append(r["token_ids"][:token_cap])
                if len(buf_ids) >= batch_size:
                    yield buf_ids, buf_tok
                    buf_ids, buf_tok = [], []
        logger.info(f"  streamed {os.path.basename(fp)}")
    if buf_ids:
        yield buf_ids, buf_tok
