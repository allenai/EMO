"""
One-off backfill: add source_index/sources to doc_ids-*.npz shard files produced by the
initial extract_doc_window.py, which recorded only (jsonl file_index, doc_start_offset)
and dropped each doc's true source_path (the S3 token file). Embeddings are untouched --
this re-derives provenance from the extraction JSONLs on CPU (one pass over the files,
regex field extraction, no token parsing; parallel over files).

    PYTHONPATH=.:src python -m src.scripts.clustering.backfill_doc_sources \\
        --docs-glob 'modular_extension/data/<run>_100B-110B/docs-*.jsonl.gz' \\
        --embeddings-dir modular_extension/cluster/emo100b_step23842/embeddings
"""

import argparse
import glob as globmod
import gzip
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SRC_RE = re.compile(r'"source_path": "([^"]+)", "doc_start_offset": (\d+)')


def file_sources(path):
    """List of (source_path, doc_start_offset) per line of one jsonl.gz."""
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            m = SRC_RE.search(line)
            assert m, f"no source_path/doc_start_offset in line of {path}"
            out.append((m.group(1), int(m.group(2))))
    logger.info(f"{path}: {len(out):,} lines")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs-glob", required=True)
    p.add_argument("--embeddings-dir", required=True)
    args = p.parse_args()

    files = sorted(globmod.glob(args.docs_glob))
    assert files
    with ThreadPoolExecutor(max_workers=len(files)) as pool:
        per_file = list(pool.map(file_sources, files))
    # global line number -> source_path, in the extractor's enumeration order
    all_sources = [s for fs in per_file for s in fs]
    logger.info(f"{len(all_sources):,} docs enumerated across {len(files)} files")

    for ids_path in sorted(globmod.glob(f"{args.embeddings_dir}/doc_ids-*.npz")):
        d = dict(np.load(ids_path, allow_pickle=True))
        if "source_index" in d:
            logger.info(f"{ids_path}: already has source_index -- skipping")
            continue
        src_table: dict = {}
        idxs = []
        for g, off in zip(d["global_doc_index"], d["doc_start_offset"]):
            sp, line_off = all_sources[int(g)]
            assert line_off == int(off), (
                f"{ids_path}: doc {g} offset mismatch (jsonl {line_off} vs npz {off}) -- "
                f"enumeration order differs, refusing to backfill"
            )
            idxs.append(src_table.setdefault(sp, len(src_table)))
        sources = [None] * len(src_table)
        for sp, i in src_table.items():
            sources[i] = sp
        d["source_index"] = np.asarray(idxs, dtype=np.int32)
        d["sources"] = np.asarray(sources)  # fixed-width str dtype loads without pickle
        np.savez(ids_path, **d)
        logger.info(f"{ids_path}: backfilled {len(idxs):,} docs, {len(sources)} distinct sources")


if __name__ == "__main__":
    main()
