#!/usr/bin/env python3
"""Stage 0 of olmoe3_squares: split the EMO 512e model's experts into k block-groups.

For MoE layers 2..9, spectral-cluster the experts into k blocks from the token-level lift graph of the
unrestricted routing pass (as in olmoe3_routing/k_sweep_partition.py), then align every layer's
blocks to the layer-9 blocks with the Q2 rule (greedy match of the document contingency table:
each document is assigned per layer to the block receiving most of its routing). Layer 1 is left
untouched: its blocks show no document-level agreement, so every group keeps all of layer 1's
experts. Output: groups.json with, per layer, the global expert ids of each group, plus the
document-group preview statistics used in the report.

Usage: python partition.py <cond_dir> --out sparse_experts/olmoe3_squares/groups.json [--k 4]
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location("coact", Path(__file__).resolve().parents[1] / "coactivation" / "analyze_coactivation.py")
coact = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(coact)
LAYERS = list(range(1, 10))
PARTITIONED = list(range(2, 10))  # layer 1 keeps all experts in every group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--min-doc-tokens", type=int, default=64)
    ap.add_argument("--size-normalized", action="store_true", help="assign documents by routing mass PER EXPERT of the group (mass / group size) instead of raw mass; needed when blocks are uneven and routing is token-level (standard MoE), where raw mass just picks the biggest block")
    a = ap.parse_args(); k = a.k
    c = np.load(a.cond / "counts.npz"); Ct = c["coact_tok"].astype(float); N = float(c["n_tok"]); E = Ct.shape[1]
    du = np.load(a.cond / "doc_usage.npy"); dl = np.load(a.cond / "doc_stats.npz")["doc_len"]; keep = dl >= a.min_doc_tokens
    labs, cl = {}, {}
    for l in PARTITIONED:
        li = LAYERS.index(l)
        lift, _ = coact.lift_matrix(Ct[li], N); W = np.log(np.maximum(lift, 1e-9)); W[W < 0] = 0; np.fill_diagonal(W, 0)
        lab = coact.spectral_labels(W, k); labs[l] = lab
        oh = np.zeros((E, k)); oh[np.arange(E), lab] = 1; cl[l] = (du[keep, li].astype(float) @ oh).argmax(1)
    ref = cl[9]; groups, agreement = {}, {}
    for l in PARTITIONED:
        J = np.zeros((k, k))
        for i in range(k): J[i] = np.bincount(ref[cl[l] == i], minlength=k)
        Jn = J / np.maximum(J.sum(1, keepdims=True), 1); perm = -np.ones(k, int); used = set()
        for _ in range(k):
            _, r, cc = max((Jn[r, cc], r, cc) for r in range(k) for cc in range(k) if perm[r] < 0 and cc not in used)
            perm[r] = cc; used.add(cc)
        g = perm[labs[l]]; groups[l] = [np.where(g == j)[0].tolist() for j in range(k)]
        agreement[l] = float(sum(J[r, perm[r]] for r in range(k)) / J.sum())
    groups[1] = [list(range(E)) for _ in range(k)]
    # document-group preview on the routing sample (layers 2..9 only, as in the assignment pass)
    tot = np.zeros((int(keep.sum()), k))
    for l in PARTITIONED:
        oh = np.zeros((E, k))
        for j in range(k): oh[groups[l][j], j] = 1
        tot += du[keep, LAYERS.index(l)].astype(float) @ oh
    share = tot.max(1) / np.maximum(tot.sum(1), 1)
    if a.size_normalized:
        n_exp = np.array([[len(groups[l][j]) for j in range(k)] for l in PARTITIONED]).sum(0)
        dg = (tot / n_exp).argmax(1); share = tot[np.arange(len(dg)), dg] / np.maximum(tot.sum(1), 1)
    else:
        dg = tot.argmax(1)
    out = dict(k=k, num_experts=E, partitioned_layers=PARTITIONED, untouched_layers=[1], source=str(a.cond), size_normalized=bool(a.size_normalized),
               groups={str(l): groups[l] for l in LAYERS}, sizes={str(l): [len(x) for x in groups[l]] for l in LAYERS},
               layer9_agreement={str(l): round(agreement[l], 3) for l in PARTITIONED},
               preview=dict(n_docs=int(keep.sum()), docs_per_group=np.bincount(dg, minlength=k).tolist(),
                            token_share=(np.bincount(dg, weights=dl[keep], minlength=k) / dl[keep].sum()).round(4).tolist(),
                            in_group_share_mean=float(share.mean()), in_group_share_median=float(np.median(share)),
                            in_group_share_pct=[float(np.percentile(share, p)) for p in (10, 25, 50, 75, 90)]))
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"))
    for l in LAYERS: print(f"layer {l}: sizes {out['sizes'][str(l)]}" + (f"  agreement with L9 groups {agreement[l]:.2f}" if l in agreement else "  (untouched)"))
    print("preview:", json.dumps(out["preview"]))


if __name__ == "__main__":
    main()
