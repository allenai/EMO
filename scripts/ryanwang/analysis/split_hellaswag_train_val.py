#!/usr/bin/env python3
"""Split HellaSwag embeddings into train+val subset for clustering.

Creates a train_val/ subdirectory containing only train+validation embeddings
(excluding test), so clustering operates only on train+val data. Test embeddings
are kept in the parent directory for later nearest-cluster assignment.

Usage:
    python scripts/ryanwang/analysis/split_hellaswag_train_val.py [DATA_DIR]

    DATA_DIR defaults to:
      claude_outputs/analysis/router_clustering_hellaswag/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301
"""

import gzip
import json
import os
import sys

import numpy as np

DEFAULT_DATA_DIR = (
    "claude_outputs/analysis/router_clustering_hellaswag/"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301"
)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR

    # Read metadata to find train+val indices
    meta = []
    with gzip.open(f"{base}/metadata.jsonl.gz", "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    train_val_indices = [i for i, m in enumerate(meta) if m["source"] in ("train", "validation")]
    test_indices = [i for i, m in enumerate(meta) if m["source"] == "test"]
    print(f"train+val: {len(train_val_indices)}, test: {len(test_indices)}, total: {len(meta)}")

    with open(f"{base}/info.json") as f:
        info = json.load(f)

    tv_dir = f"{base}/train_val"
    os.makedirs(tv_dir, exist_ok=True)

    # Filter embeddings (both dense and sparse)
    emb_files = [f for f in os.listdir(base) if f.startswith("embeddings_") and f.endswith(".npy")]
    for emb_file in sorted(emb_files):
        emb = np.load(f"{base}/{emb_file}")
        tv_emb = emb[train_val_indices]
        np.save(f"{tv_dir}/{emb_file}", tv_emb)
        print(f"  {emb_file}: {emb.shape} -> {tv_emb.shape}")

    # Filter metadata
    tv_meta = [meta[i] for i in train_val_indices]
    with gzip.open(f"{tv_dir}/metadata.jsonl.gz", "wt") as f:
        for m in tv_meta:
            f.write(json.dumps(m) + "\n")

    # Save info.json
    tv_info = dict(info)
    tv_info["num_docs"] = len(train_val_indices)
    tv_info["total_tokens"] = sum(m["doc_len"] for m in tv_meta)
    tv_info["source_doc_counts"] = {
        "train": sum(1 for m in tv_meta if m["source"] == "train"),
        "validation": sum(1 for m in tv_meta if m["source"] == "validation"),
    }
    tv_info["note"] = "Filtered to train+validation only (test excluded for clustering)"
    with open(f"{tv_dir}/info.json", "w") as f:
        json.dump(tv_info, f, indent=2)

    print(f"\nSaved train+val subset to {tv_dir}/")


if __name__ == "__main__":
    main()
