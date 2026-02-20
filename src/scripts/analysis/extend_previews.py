"""
Re-stream the same S3 documents (no GPU needed) to extract full document text,
then update metadata.jsonl.gz, regenerate cluster summary.json, and rebuild
the HTML visualizer with full-length previews.

The extraction in extract_router_embeddings.py is fully deterministic (sequential
S3 reads, no randomization), so replaying it yields the same documents in order.

Usage:
    python -m src.scripts.analysis.extend_previews \
        --output-dir claude_outputs/analysis/router_clustering \
        --k 64 \
        --max-preview-chars 3000
"""

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 100257
BYTES_PER_TOKEN = 4


def stream_bytes_from_s3(s3_path: str, num_bytes: int) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["aws", "s3api", "get-object",
             "--bucket", "ai2-llm",
             "--key", s3_path.replace("s3://ai2-llm/", ""),
             "--range", f"bytes=0-{num_bytes - 1}",
             tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"S3 range-GET failed: {result.stderr[:200]}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def tokens_from_bytes(raw: bytes) -> np.ndarray:
    n = len(raw) // BYTES_PER_TOKEN
    return np.frombuffer(raw[:n * BYTES_PER_TOKEN], dtype=np.uint32).astype(np.int32)


def iter_documents(tokens, min_len=32, max_len=2048):
    eos_pos = np.where(tokens == EOS_TOKEN_ID)[0]
    start = 0
    for pos in eos_pos:
        doc = tokens[start:pos + 1]
        if min_len <= len(doc) <= max_len:
            yield doc
        start = pos + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="claude_outputs/analysis/router_clustering",
                        help="Dir with shared data files (metadata.jsonl.gz, mix_composition.json, info.json)")
    parser.add_argument("--analysis-dir", default=None,
                        help="Dir with clusters_k{k}/ and where HTML/report go. Defaults to --data-dir.")
    parser.add_argument("--emb-file", default=None,
                        help="Embedding .npy path. Defaults to <data-dir>/embeddings_optA_avgprob.npy.")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--max-preview-chars", type=int, default=3000)
    parser.add_argument("--model-path",
                        default="models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir or args.data_dir
    cluster_dir = os.path.join(analysis_dir, f"clusters_k{args.k}")
    meta_path   = os.path.join(args.data_dir, "metadata.jsonl.gz")
    comp_path   = os.path.join(args.data_dir, "mix_composition.json")
    info_path   = os.path.join(args.data_dir, "info.json")

    # ── Load existing state ──────────────────────────────────────────────────
    logger.info("Loading existing metadata...")
    old_meta = []
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            old_meta.append(json.loads(line))
    logger.info(f"  {len(old_meta)} documents")

    with open(comp_path) as f:
        composition = json.load(f)
    with open(info_path) as f:
        info = json.load(f)

    # ── Load tokenizer (no model weights) ───────────────────────────────────
    logger.info(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    total_tokens = info["total_tokens"]

    # ── Replay S3 streaming in the same order as the original extraction ─────
    # Original extraction iterates sources sorted by -fraction, sequentially
    # reading from the start of each S3 file. No randomness involved.
    logger.info("Replaying S3 streaming (no GPU) to get full document text...")
    new_previews_by_source = {}  # label → [(doc_len, text), ...]

    for label, src_info in sorted(composition["sources"].items(), key=lambda x: -x[1]["fraction"]):
        target_source_tokens = int(src_info["fraction"] * total_tokens)
        if target_source_tokens < 32:
            continue

        bytes_needed = int(target_source_tokens * BYTES_PER_TOKEN * 1.5)
        texts = []
        collected = 0

        for s3_path in src_info["all_files"]:
            if collected >= target_source_tokens:
                break
            remaining = max(0, bytes_needed - collected * BYTES_PER_TOKEN)
            logger.info(f"  [{label}] {remaining/1e6:.1f} MB from {s3_path.split('/')[-1]} ...")
            try:
                raw = stream_bytes_from_s3(s3_path, remaining)
            except Exception as e:
                logger.warning(f"  Skipping {s3_path}: {e}")
                continue

            toks = tokens_from_bytes(raw)
            for doc in iter_documents(toks):
                text = tokenizer.decode(doc.tolist(), skip_special_tokens=True)
                texts.append((len(doc), text[:args.max_preview_chars]))
                collected += len(doc)
                if collected >= target_source_tokens:
                    break

        new_previews_by_source[label] = texts
        logger.info(f"  [{label}] {len(texts)} docs")

    # ── Match replayed text to existing metadata ─────────────────────────────
    logger.info("Matching replayed text to existing metadata...")
    source_cursors = {label: 0 for label in new_previews_by_source}
    new_meta = []
    mismatches = 0

    for m in old_meta:
        label = m["source"]
        cursor = source_cursors.get(label, 0)
        texts = new_previews_by_source.get(label, [])
        if cursor < len(texts):
            replayed_len, text = texts[cursor]
            if replayed_len != m["doc_len"]:
                mismatches += 1
                text = m["preview"]   # fallback: keep old short preview
            source_cursors[label] = cursor + 1
        else:
            text = m["preview"]
        new_meta.append({"source": m["source"], "doc_len": m["doc_len"], "preview": text})

    if mismatches:
        logger.warning(f"  {mismatches} doc_len mismatches — those entries kept old preview")
    else:
        logger.info("  All doc_lens matched ✓")

    # ── Save updated metadata ────────────────────────────────────────────────
    logger.info(f"Saving updated metadata (up to {args.max_preview_chars}-char previews)...")
    with gzip.open(meta_path, "wt") as f:
        for m in new_meta:
            f.write(json.dumps(m) + "\n")
    logger.info(f"  Saved → {meta_path}")

    # ── Update summary.json rep-doc previews ─────────────────────────────────
    # The cluster assignments array maps global doc index → cluster id.
    # Use it to look up the new preview for each representative doc.
    summary_path = os.path.join(cluster_dir, "summary.json")
    assignments_path = os.path.join(cluster_dir, "assignments.npy")
    if os.path.exists(summary_path) and os.path.exists(assignments_path):
        logger.info("Updating summary.json with full rep-doc previews...")
        with open(summary_path) as f:
            summary = json.load(f)
        labels = np.load(assignments_path)

        # Build per-cluster list of global indices (preserving order)
        cluster_indices = {}
        for global_idx, lbl in enumerate(labels):
            cluster_indices.setdefault(int(lbl), []).append(global_idx)

        for c in summary:
            cid = c["cluster"]
            indices = cluster_indices.get(cid, [])
            # Each rep_doc was the k-th nearest to the centroid; we stored
            # them in order — match by position within representative_docs
            # using source+doc_len to identify the global index.
            for rep in c["representative_docs"]:
                # Find the first cluster member with matching source & doc_len
                for gidx in indices:
                    m = new_meta[gidx]
                    if m["source"] == rep["source"] and m["doc_len"] == rep["doc_len"]:
                        rep["preview"] = m["preview"]
                        break

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Saved → {summary_path}")

    # ── Regenerate HTML visualizer ───────────────────────────────────────────
    logger.info("Regenerating HTML visualizer with full previews...")
    emb_file = args.emb_file or os.path.join(args.data_dir, "embeddings_optA_avgprob.npy")
    sys.argv = [
        "generate_cluster_viz",
        "--output-dir", analysis_dir,
        "--data-dir", args.data_dir,
        "--emb-file", emb_file,
        "--k", str(args.k),
    ]
    import importlib
    import src.scripts.analysis.generate_cluster_viz as gen
    importlib.reload(gen)
    gen.main()

    logger.info("All done. Full-text previews now embedded in cluster_explorer.html")


if __name__ == "__main__":
    main()
