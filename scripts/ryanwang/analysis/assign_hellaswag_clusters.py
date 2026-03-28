#!/usr/bin/env python3
"""Fit spherical k-means on HellaSwag train+val embeddings and assign test examples.

Reproduces the mean_pca_l2 + spherical_kmeans pipeline from transform_and_cluster.py,
but with separate fit/transform steps so test examples can be projected into the same
space and assigned to their nearest cluster.

Note: We can't reuse transform_and_cluster.py's functions directly because they
combine fit+transform in one call (no way to apply a fitted PCA to new data) and
don't expose cluster centroids.

Outputs a JSON file with per-split cluster assignments that the task classes in
splits_hellaswag.py can load to create per-cluster HellaSwag tasks.

Usage:
    python scripts/ryanwang/analysis/assign_hellaswag_clusters.py [DATA_DIR] [--k 6]

    DATA_DIR defaults to:
      claude_outputs/analysis/router_clustering_hellaswag/
      twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301
"""

import argparse
import gzip
import json
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

DEFAULT_DATA_DIR = (
    "claude_outputs/analysis/router_clustering_hellaswag/"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301"
)


def main():
    parser = argparse.ArgumentParser(description="Assign HellaSwag examples to clusters")
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--embedding", default="topk_freq")
    args = parser.parse_args()

    data_dir = args.data_dir
    tv_dir = os.path.join(data_dir, "train_val")
    out_dir = os.path.join(data_dir, "cluster_assignments")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading train+val embeddings from {tv_dir}...")
    tv_emb = np.load(os.path.join(tv_dir, f"embeddings_{args.embedding}.npy")).astype(np.float32)
    print(f"  shape: {tv_emb.shape}")

    print("Loading full metadata...")
    meta = []
    with gzip.open(os.path.join(data_dir, "metadata.jsonl.gz"), "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    print(f"Loading full embeddings to extract test subset...")
    full_emb = np.load(os.path.join(data_dir, f"embeddings_{args.embedding}.npy")).astype(np.float32)
    test_mask = [m["source"] == "test" for m in meta]
    test_emb = full_emb[test_mask]
    print(f"  test embeddings: {test_emb.shape}")

    tv_meta = []
    with gzip.open(os.path.join(tv_dir, "metadata.jsonl.gz"), "rt") as f:
        for line in f:
            tv_meta.append(json.loads(line))

    # ── Fit PCA on train+val (same as mean_pca_l2 in transform_and_cluster.py) ──
    print("\nFitting mean_pca_l2 on train+val...")
    mean = tv_emb.mean(axis=0, keepdims=True)
    centered = tv_emb - mean

    n_components = min(centered.shape[0], centered.shape[1])
    pca_full = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca_full.fit(centered)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_pca = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f"  PCA: {n_pca} components explain {cumvar[n_pca-1]:.1%} variance")

    pca = PCA(n_components=n_pca, random_state=42)
    tv_reduced = pca.fit_transform(centered)
    tv_normed = normalize(tv_reduced, norm="l2")

    # ── Fit spherical k-means on train+val ───────────────────────────────
    print(f"\nFitting spherical k-means (k={args.k}) on train+val...")
    km = MiniBatchKMeans(n_clusters=args.k, n_init=10, max_iter=500, batch_size=4096, random_state=42)
    km.fit(tv_normed)

    for iteration in range(20):
        old_centers = km.cluster_centers_.copy()
        km.cluster_centers_ = normalize(km.cluster_centers_, norm="l2")
        tv_labels = km.predict(tv_normed)
        for c in range(args.k):
            mask = tv_labels == c
            if mask.sum() > 0:
                km.cluster_centers_[c] = normalize(
                    tv_normed[mask].mean(axis=0, keepdims=True), norm="l2"
                )[0]
        delta = np.abs(km.cluster_centers_ - old_centers).max()
        if delta < 1e-6:
            print(f"  Converged at iteration {iteration + 1}")
            break

    tv_labels = km.predict(tv_normed)
    centroids = km.cluster_centers_
    print(f"  Spherical KMeans: k={args.k}, {iteration + 1} refinement iterations")

    # ── Transform test + assign to nearest cluster ───────────────────────
    print("\nAssigning test examples to nearest cluster...")
    test_centered = test_emb - mean
    test_reduced = pca.transform(test_centered)
    test_normed = normalize(test_reduced, norm="l2")
    test_labels = np.argmax(test_normed @ centroids.T, axis=1)

    # ── Build per-cluster, per-split index lists ─────────────────────────
    # Indices are positions within the train_val array (for train/val)
    # and positions within the test-only portion (for test)
    assignments = {}
    print(f"\nCluster sizes:")
    print(f"  {'Cluster':<10} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for c in range(args.k):
        c_train = [i for i, m in enumerate(tv_meta) if tv_labels[i] == c and m["source"] == "train"]
        c_val = [i for i, m in enumerate(tv_meta) if tv_labels[i] == c and m["source"] == "validation"]
        c_test = [i for i in range(len(test_labels)) if test_labels[i] == c]

        assignments[str(c)] = {
            "train": c_train,
            "validation": c_val,
            "test": c_test,
        }
        total = len(c_train) + len(c_val) + len(c_test)
        print(f"  {c:<10} {len(c_train):>8} {len(c_val):>8} {len(c_test):>8} {total:>8}")

    total_train = sum(len(v["train"]) for v in assignments.values())
    total_val = sum(len(v["validation"]) for v in assignments.values())
    total_test = sum(len(v["test"]) for v in assignments.values())
    print(f"  {'TOTAL':<10} {total_train:>8} {total_val:>8} {total_test:>8} {total_train+total_val+total_test:>8}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "k": args.k,
        "embedding": args.embedding,
        "transform": "mean_pca_l2",
        "cluster_method": "spherical_kmeans",
        "pca_components": n_pca,
        "data_dir": data_dir,
        "note": (
            "Indices in 'train' and 'validation' are positions within the train_val "
            "embedding array (ordered: all train then all validation, matching "
            "train_val/metadata.jsonl.gz). Indices in 'test' are positions within "
            "the test-only subset of the full embedding array."
        ),
        "clusters": assignments,
    }

    k_suffix = f"_k{args.k}"
    assignments_file = os.path.join(out_dir, f"cluster_assignments{k_suffix}.json")
    with open(assignments_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {assignments_file}")

    np.save(os.path.join(out_dir, f"train_val_assignments{k_suffix}.npy"), tv_labels)
    np.save(os.path.join(out_dir, f"test_assignments{k_suffix}.npy"), test_labels)
    np.save(os.path.join(out_dir, f"centroids{k_suffix}.npy"), centroids)
    print(f"Saved: train_val_assignments{k_suffix}.npy ({tv_labels.shape})")
    print(f"Saved: test_assignments{k_suffix}.npy ({test_labels.shape})")
    print(f"Saved: centroids{k_suffix}.npy ({centroids.shape})")


if __name__ == "__main__":
    main()
