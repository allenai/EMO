#!/usr/bin/env python3
"""K-means expert partition (user request 2026-11-15): in every partitioned layer (2-9) the experts are clustered with k-means on their
router weight rows (the vector a token's hidden state is scored against; rows L2-normalised, so the clustering is by direction), giving k
groups of UNEQUAL size; layer 1 stays whole in every group, as in partition.py / partition_random.py. The clusters of layer 9 are numbered
by size (largest first); every lower layer's clusters are matched one-to-one to the layer above by centroid cosine similarity (Hungarian
assignment; the routers of all layers read the same residual stream), so group g holds related experts across layers. --min-size: a
cluster smaller than this (a sub-model needs at least top_k = 16 experts per layer) is refilled with the nearest experts of other clusters
that can spare them. Writes the groups.json layout of partition_random.py.
  python partition_kmeans.py --checkpoint <full step dir> --out <groups.json> [--k 8] [--seed 0] [--min-size 24]"""
import argparse, json
from pathlib import Path
import numpy as np

LAYERS = list(range(1, 10)); PARTITIONED = list(range(2, 10))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-size", type=int, default=24); a = ap.parse_args()
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment
    from olmo_core.distributed.checkpoint import load_keys
    src = str(a.checkpoint / "model_and_optim"); k = a.k; X, lab, cen = {}, {}, {}
    for l in PARTITIONED:
        w = next(load_keys(src, [f"module.blocks.{l}.routed_experts_router.weight.main"])).float().numpy()
        E = w.size // json.load(open(a.checkpoint / "config.json"))["model"]["d_model"]; w = w.reshape(E, -1)
        X[l] = w / np.linalg.norm(w, axis=1, keepdims=True)
        km = KMeans(n_clusters=k, n_init=50, random_state=a.seed).fit(X[l]); lb = km.labels_.copy(); moved = 0
        while True:   # refill clusters below the minimum size
            sz = np.bincount(lb, minlength=k); j = int(sz.argmin())
            if sz[j] >= a.min_size: break
            c = X[l][lb == j].mean(0); cand = np.where(sz[lb] > a.min_size)[0]; i = cand[(X[l][cand] @ c).argmax()]; lb[i] = j; moved += 1
        lab[l] = lb; cen[l] = np.stack([X[l][lb == j].mean(0) for j in range(k)]); cen[l] /= np.linalg.norm(cen[l], axis=1, keepdims=True)
        print(f"layer {l}: k-means sizes {sorted(np.bincount(km.labels_, minlength=k).tolist(), reverse=True)}, {moved} experts moved for the minimum size, inertia {km.inertia_:.1f}")
    order = np.argsort(-np.bincount(lab[9], minlength=k), kind="stable"); perm = {9: np.empty(k, int)}; perm[9][order] = np.arange(k); sim = {}
    for l in range(8, 1, -1):   # cluster i of layer l -> the group of its matched cluster in layer l+1
        Sm = cen[l] @ cen[l + 1].T; r, c = linear_sum_assignment(-Sm); perm[l] = np.empty(k, int); perm[l][r] = perm[l + 1][c]; sim[l] = float(Sm[r, c].mean())
    groups = {1: [list(range(E)) for _ in range(k)]}
    for l in PARTITIONED:
        g = perm[l][lab[l]]; groups[l] = [np.where(g == j)[0].tolist() for j in range(k)]
    out = dict(k=k, num_experts=E, partitioned_layers=PARTITIONED, untouched_layers=[1], source=f"kmeans(router rows, L2-normalised, seed={a.seed}, min_size={a.min_size}) of {a.checkpoint}",
               size_normalized=False, groups={str(l): groups[l] for l in LAYERS}, sizes={str(l): [len(x) for x in groups[l]] for l in LAYERS}, layer9_agreement={}, preview=None,
               matched_centroid_cosine={str(l): round(sim[l], 3) for l in sim})
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"))
    for l in PARTITIONED: print(f"layer {l}: group sizes {out['sizes'][str(l)]}" + (f"  mean matched centroid cosine to layer {l + 1}: {sim[l]:.3f}" if l in sim else ""))
    tot = np.array([out["sizes"][str(l)] for l in PARTITIONED]).sum(0); print("experts per group over layers 2-9:", tot.tolist())


if __name__ == "__main__":
    main()
