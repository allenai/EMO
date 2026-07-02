"""
Export the document -> cluster partition as a provenance-keyed JSONL.

Joins the clustering result (assignments.npy, row-aligned with the merged data dir) with
doc_ids.npz to produce one line per document:
    {"source_path", "doc_start_offset", "doc_len", "cluster"}
keyed exactly like the extraction JSONLs (so the partition can be joined back onto the
token data), plus a summary JSON with per-cluster doc/token counts and top sources.

Usage:
    PYTHONPATH=.:src python -m src.scripts.clustering.export_doc_partition \\
        --data-dir modular_extension/cluster/emo100b_step23842 \\
        --result-dir <data-dir>/doc_probs_mean_pca_l2_spherical_kmeans_k64 \\
        --output-prefix modular_extension/data/<run>_100B-110B/doc_clusters_k64
"""

import argparse
import gzip
import json
import logging
import os
from collections import Counter, defaultdict

import numpy as np

from .build_doc_window_datadir import source_from_path

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--output-prefix", required=True,
                   help="writes <prefix>.jsonl.gz and <prefix>_summary.json")
    args = p.parse_args()

    labels = np.load(os.path.join(args.result_dir, "assignments.npy"))
    ids = np.load(os.path.join(args.data_dir, "doc_ids.npz"), allow_pickle=False)
    files = [str(x) for x in np.load(os.path.join(args.data_dir, "doc_ids.npz"), allow_pickle=True)["files"]]
    info = json.load(open(os.path.join(args.data_dir, "info.json")))

    n = len(labels)
    assert n == len(ids["doc_start_offset"]) == info["num_docs"], "row misalignment"

    fi = ids["file_index"]
    off = ids["doc_start_offset"]
    dl = ids["doc_len"]

    out_jsonl = f"{args.output_prefix}.jsonl.gz"
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    doc_counts = Counter()
    token_counts = defaultdict(int)
    src_counts = defaultdict(Counter)
    with gzip.open(out_jsonl, "wt") as f:
        for i in range(n):
            c = int(labels[i])
            f.write(json.dumps({
                "source_path": files[fi[i]],
                "doc_start_offset": int(off[i]),
                "doc_len": int(dl[i]),
                "cluster": c,
            }) + "\n")
            doc_counts[c] += 1
            token_counts[c] += int(dl[i])
            src_counts[c][source_from_path(files[fi[i]])] += 1

    k = int(labels.max()) + 1
    summary = {
        "data_dir": args.data_dir,
        "result_dir": args.result_dir,
        "num_docs": n,
        "total_doc_tokens": int(dl.sum()),
        "k": k,
        "clusters": [
            {
                "cluster": c,
                "num_docs": doc_counts.get(c, 0),
                "num_tokens": token_counts.get(c, 0),
                "top_sources": dict(src_counts[c].most_common(5)),
            }
            for c in range(k)
        ],
    }
    with open(f"{args.output_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    sizes = sorted(doc_counts.values())
    logger.info(
        f"DONE: {n:,} docs, {int(dl.sum()):,} tokens -> {out_jsonl} | k={k}, "
        f"cluster docs min/median/max = {sizes[0]:,}/{sizes[len(sizes)//2]:,}/{sizes[-1]:,}"
    )


if __name__ == "__main__":
    main()
