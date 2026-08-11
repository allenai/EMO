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
    p.add_argument("--embeddings-dir", required=True,
                   help="one dir, or a comma-separated list to merge several doc windows "
                        "(e.g. embeddings,embeddings_110B-130B) into one joint data dir")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--shards", required=True, help="e.g. '0-15' or '0-127' or '0,3,5'")
    p.add_argument("--num-shards", type=int, default=128)
    p.add_argument("--expect-docs", type=int, default=None,
                   help="assert the merged row count equals this (full-sweep check)")
    args = p.parse_args()

    shards = parse_shards(args.shards)
    embed_dirs = args.embeddings_dir.split(",")

    # Row order is window-major, then shard-major: window 0's shard rows, then window 1's.
    # `window_index` records each row's window. Note global_doc_index and file_index are
    # RELATIVE TO THEIR WINDOW (each window has its own doc enumeration and file list);
    # the window-agnostic join key is (source_paths[source_index], doc_start_offset).
    # The same physical document can legitimately appear in several windows (a long doc's
    # shuffled chunks can straddle the window boundary), so cross-window duplicates by
    # that key are expected and kept.
    probs_list, topk_list, ids_list, infos, win_of = [], [], [], [], []
    files_per_window = []
    for w, ed in enumerate(embed_dirs):
        win_infos = []
        for s in shards:
            paths = {
                "probs": os.path.join(ed, f"doc_probs-{s:03d}.npy"),
                "topk": os.path.join(ed, f"doc_topk_freq-{s:03d}.npy"),
                "ids": os.path.join(ed, f"doc_ids-{s:03d}.npz"),
                "info": os.path.join(ed, f"info-{s:03d}.json"),
            }
            missing = [k for k, v in paths.items() if not os.path.exists(v)]
            assert not missing, f"window {w} shard {s} missing outputs: {missing}"
            pr = np.load(paths["probs"])
            tk = np.load(paths["topk"])
            ids = np.load(paths["ids"], allow_pickle=True)  # sources may be object-dtype
            info = json.load(open(paths["info"]))
            assert pr.shape == tk.shape and pr.shape[0] == len(ids["global_doc_index"]) == info["num_docs"], \
                f"window {w} shard {s}: inconsistent row counts"
            assert np.isfinite(pr.astype(np.float32)).all(), f"window {w} shard {s}: non-finite doc_probs"
            assert info["num_shards"] == args.num_shards, f"window {w} shard {s}: num_shards mismatch"
            probs_list.append(pr)
            topk_list.append(tk)
            ids_list.append(ids)
            win_infos.append(info)
            win_of.append(np.full(pr.shape[0], w, dtype=np.int8))
            logger.info(f"window {w} shard {s}: {pr.shape[0]:,} docs ok")
        # per-window duplicate check (global_doc_index is window-scoped)
        win_gdi = np.concatenate([ids["global_doc_index"] for ids in ids_list[-len(shards):]])
        assert len(np.unique(win_gdi)) == len(win_gdi), f"window {w}: duplicate docs across shards"
        files = win_infos[0]["files"]
        assert all(i["files"] == files for i in win_infos), f"window {w}: shards saw different file lists"
        files_per_window.append(files)
        infos.extend(win_infos)

    probs = np.concatenate(probs_list)
    topk = np.concatenate(topk_list)
    merged_ids = {
        k: np.concatenate([ids[k] for ids in ids_list])
        for k in ("global_doc_index", "file_index", "doc_start_offset", "doc_len", "n_embed_tokens")
    }
    merged_ids["window_index"] = np.concatenate(win_of)
    # Re-key each shard's compact source table into one global table of true source paths
    # (the S3 token files). Shards lacking source_index need backfill_doc_sources.py first.
    global_table: dict = {}
    src_idx_parts = []
    for ids in ids_list:
        assert "source_index" in ids, (
            "a shard's doc_ids lacks source_index -- run backfill_doc_sources.py "
            "(or re-extract with the current extractor)"
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
    if args.expect_docs is not None:
        assert n == args.expect_docs, f"expected {args.expect_docs:,} docs, got {n:,}"

    files = files_per_window[0] if len(embed_dirs) == 1 else None
    consistent = ("model_path", "max_tokens_per_doc", "num_layers", "num_experts",
                  "num_shared_experts", "num_standard_experts", "top_k", "routed_top_k", "emb_dim")
    for k in consistent:
        assert all(i[k] == infos[0][k] for i in infos), f"windows disagree on {k}"
    sources = [source_from_path(sp) for sp in source_paths]  # compact mix-source labels

    os.makedirs(args.data_dir, exist_ok=True)
    # cluster.py caches preprocessed_*.npy keyed by embedding/preprocess name only; a
    # re-merge with a different shard set would silently reuse stale caches. Invalidate.
    for stale in glob.glob(os.path.join(args.data_dir, "preprocessed_*.npy")):
        logger.info(f"removing stale preprocess cache {stale}")
        os.remove(stale)
    np.save(os.path.join(args.data_dir, "embeddings_doc_probs.npy"), probs)
    np.save(os.path.join(args.data_dir, "embeddings_doc_topk_freq.npy"), topk)
    extra = {"files": np.array(files)} if files is not None else {}
    np.savez(os.path.join(args.data_dir, "doc_ids.npz"), **merged_ids,
             source_paths=np.asarray(source_paths), **extra)

    with gzip.open(os.path.join(args.data_dir, "metadata_docs.jsonl.gz"), "wt") as f:
        si = merged_ids["source_index"]
        dl = merged_ids["doc_len"]
        for i in range(n):
            f.write(json.dumps({
                "doc_index": i,
                "source": sources[si[i]],
                "doc_len": int(dl[i]),
            }) + "\n")

    window_globs = []
    for i in infos:
        if i["docs_glob"] not in window_globs:
            window_globs.append(i["docs_glob"])
    info = {
        "kind": "doc_window_router_embeddings",
        "model_path": infos[0]["model_path"],
        "docs_glob": window_globs[0] if len(window_globs) == 1 else window_globs,
        "num_windows": len(embed_dirs),
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
