"""
Merge extract_doc_window.py shard outputs into a cluster.py-compatible data dir.

Produces in --data-dir:
  embeddings_doc_probs.npy      (N, emb_dim) float16
  embeddings_doc_topk_freq.npy  (N, emb_dim) float16
  doc_ids.npz                   merged id arrays (aligned row-for-row with embeddings)
  metadata_docs.jsonl.gz        per-row {doc_index, source, doc_len} (source = mix source
                                name parsed from the file path, for cluster.py summaries)
  info.json                     moe config + counts + shard provenance

Rows are ordered shard-major (shard 0's docs, then shard 1's, ...); within a shard docs
follow the global enumeration. doc_ids.npz carries (file_index, doc_start_offset) plus
global_doc_index for exact provenance back to the extraction JSONLs.

Usage:
    PYTHONPATH=.:src python -m src.scripts.clustering.build_doc_window_datadir \\
        --embeddings-dir modular_extension/cluster/emo100b_step23842/embeddings \\
        --data-dir modular_extension/cluster/emo100b_step23842 \\
        --shards 0-15 --num-shards 128
"""

import argparse
import glob
import gzip
import json
import logging
import os
import re

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_shards(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def source_from_path(path: str) -> str:
    """Extract a compact mix-source label from a token-file S3 path."""
    # e.g. s3://ai2-llm/preprocessed/dclm/.../part-12-00000.npy -> segment after 'preprocessed'
    m = re.search(r"preprocessed/([^/]+(?:/[^/]+)?)/", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--shards", required=True, help="e.g. '0-15' or '0-127' or '0,3,5'")
    p.add_argument("--num-shards", type=int, default=128)
    p.add_argument("--expect-docs", type=int, default=None,
                   help="assert the merged row count equals this (full-sweep check)")
    args = p.parse_args()

    shards = parse_shards(args.shards)
    ed = args.embeddings_dir

    probs_list, topk_list, ids_list, infos = [], [], [], []
    for s in shards:
        paths = {
            "probs": os.path.join(ed, f"doc_probs-{s:03d}.npy"),
            "topk": os.path.join(ed, f"doc_topk_freq-{s:03d}.npy"),
            "ids": os.path.join(ed, f"doc_ids-{s:03d}.npz"),
            "info": os.path.join(ed, f"info-{s:03d}.json"),
        }
        missing = [k for k, v in paths.items() if not os.path.exists(v)]
        assert not missing, f"shard {s} missing outputs: {missing}"
        pr = np.load(paths["probs"])
        tk = np.load(paths["topk"])
        ids = np.load(paths["ids"], allow_pickle=True)  # sources may be object-dtype
        info = json.load(open(paths["info"]))
        assert pr.shape == tk.shape and pr.shape[0] == len(ids["global_doc_index"]) == info["num_docs"], \
            f"shard {s}: inconsistent row counts"
        assert np.isfinite(pr.astype(np.float32)).all(), f"shard {s}: non-finite doc_probs"
        assert info["num_shards"] == args.num_shards, f"shard {s}: num_shards mismatch"
        probs_list.append(pr)
        topk_list.append(tk)
        ids_list.append(ids)
        infos.append(info)
        logger.info(f"shard {s}: {pr.shape[0]:,} docs ok")

    probs = np.concatenate(probs_list)
    topk = np.concatenate(topk_list)
    merged_ids = {
        k: np.concatenate([ids[k] for ids in ids_list])
        for k in ("global_doc_index", "file_index", "doc_start_offset", "doc_len", "n_embed_tokens")
    }
    # Re-key each shard's compact source table into one global table of true source paths
    # (the S3 token files). Shards lacking source_index need backfill_doc_sources.py first.
    global_table: dict = {}
    src_idx_parts = []
    for s, ids in zip(shards, ids_list):
        assert "source_index" in ids, (
            f"shard {s} doc_ids lacks source_index -- run backfill_doc_sources.py "
            f"(or re-extract with the current extractor)"
        )
        shard_sources = [str(x) for x in ids["sources"]]
        remap = np.array([global_table.setdefault(sp, len(global_table)) for sp in shard_sources],
                         dtype=np.int32)
        src_idx_parts.append(remap[ids["source_index"]])
    merged_ids["source_index"] = np.concatenate(src_idx_parts)
    source_paths = [None] * len(global_table)
    for sp, i in global_table.items():
        source_paths[i] = sp

    n = probs.shape[0]
    assert len(np.unique(merged_ids["global_doc_index"])) == n, "duplicate docs across shards"
    if args.expect_docs is not None:
        assert n == args.expect_docs, f"expected {args.expect_docs:,} docs, got {n:,}"

    files = infos[0]["files"]
    assert all(i["files"] == files for i in infos), "shards saw different file lists"
    sources = [source_from_path(sp) for sp in source_paths]  # compact mix-source labels

    os.makedirs(args.data_dir, exist_ok=True)
    # cluster.py caches preprocessed_*.npy keyed by embedding/preprocess name only; a
    # re-merge with a different shard set would silently reuse stale caches. Invalidate.
    for stale in glob.glob(os.path.join(args.data_dir, "preprocessed_*.npy")):
        logger.info(f"removing stale preprocess cache {stale}")
        os.remove(stale)
    np.save(os.path.join(args.data_dir, "embeddings_doc_probs.npy"), probs)
    np.save(os.path.join(args.data_dir, "embeddings_doc_topk_freq.npy"), topk)
    np.savez(os.path.join(args.data_dir, "doc_ids.npz"), **merged_ids,
             files=np.array(files), source_paths=np.asarray(source_paths))

    with gzip.open(os.path.join(args.data_dir, "metadata_docs.jsonl.gz"), "wt") as f:
        si = merged_ids["source_index"]
        dl = merged_ids["doc_len"]
        for i in range(n):
            f.write(json.dumps({
                "doc_index": i,
                "source": sources[si[i]],
                "doc_len": int(dl[i]),
            }) + "\n")

    info = {
        "kind": "doc_window_router_embeddings",
        "model_path": infos[0]["model_path"],
        "docs_glob": infos[0]["docs_glob"],
        "max_tokens_per_doc": infos[0]["max_tokens_per_doc"],
        "num_shards": args.num_shards,
        "shards_merged": shards,
        "num_docs": int(n),
        "num_embed_tokens": int(sum(i["num_embed_tokens"] for i in infos)),
        "total_doc_tokens": int(merged_ids["doc_len"].sum()),
        **{k: infos[0][k] for k in ("num_layers", "num_experts", "num_shared_experts",
                                    "num_standard_experts", "top_k", "routed_top_k", "emb_dim")},
    }
    with open(os.path.join(args.data_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(
        f"DONE: {n:,} docs merged from {len(shards)} shards -> {args.data_dir} "
        f"({info['num_embed_tokens']:,} embed tokens, {info['total_doc_tokens']:,} doc tokens)"
    )


if __name__ == "__main__":
    main()
