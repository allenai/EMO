#!/usr/bin/env python3
"""Data for the interactive layer-i -> layer-9 block-agreement grid (report Q6).

For one routing pass and one k: spectral-cluster every MoE layer's experts into k blocks (token lift
graph, as in k_sweep_partition.py), assign each document (>= min tokens) to a block per layer with its
purity, and for every source layer i and every N in a grid, take the top-N purity documents per
layer-i block and tabulate their layer-9 blocks (k x k contingency), plus NMI, majority share and the
purity threshold that N implies. Rows of each table are permuted once (greedy match at N=1000) so the
diagonal is the natural reading order. Output: <out>/<tag>_k<k>_grid.json.

Usage: python layer_pair_grid.py <cond_dir> --tag emo1000_full --out <dir> --k 4
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location("coact", Path(__file__).resolve().parents[1] / "coactivation" / "analyze_coactivation.py")
coact = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(coact)
_spec2 = importlib.util.spec_from_file_location("ksw", Path(__file__).resolve().parent / "k_sweep_partition.py")
ksw = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(ksw)
LAYERS = list(range(1, 10))
N_GRID = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 100000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", type=Path); ap.add_argument("--tag", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--min-doc-tokens", type=int, default=64); ap.add_argument("--target", type=int, default=9)
    a = ap.parse_args(); k = a.k
    c = np.load(a.cond / "counts.npz"); Ct = c["coact_tok"].astype(float); N = float(c["n_tok"])
    du = np.load(a.cond / "doc_usage.npy"); doc_len = np.load(a.cond / "doc_stats.npz")["doc_len"]
    keep = doc_len >= a.min_doc_tokens; E = Ct.shape[1]; a.out.mkdir(parents=True, exist_ok=True)
    labs, cl, pur = [], [], []
    for li in range(len(LAYERS)):
        lift, _ = coact.lift_matrix(Ct[li], N); W = np.log(np.maximum(lift, 1e-9)); W[W < 0] = 0; np.fill_diagonal(W, 0)
        lab = coact.spectral_labels(W, k); labs.append(lab)
        onehot = np.zeros((E, k)); onehot[np.arange(E), lab] = 1
        U = du[keep, li].astype(float); mass = U @ onehot; tot = np.maximum(mass.sum(1), 1)
        cl.append(mass.argmax(1)); pur.append(mass.max(1) / tot)
    T = LAYERS.index(a.target); cb = cl[T]
    out = dict(tag=a.tag, k=k, target=a.target, n_grid=N_GRID, n_docs=int(keep.sum()), layers={})
    for li, l in enumerate(LAYERS):
        ca, pa = cl[li], pur[li]
        order = [np.where(ca == g)[0][np.argsort(-pa[ca == g])] for g in range(k)]   # docs per group, best purity first
        tables, stats = [], []
        for n in N_GRID:
            joint = np.zeros((k, k), int); kept = []
            for g in range(k):
                idx = order[g][:n]; kept.append(idx); joint[g] = np.bincount(cb[idx], minlength=k)
            kept = np.concatenate(kept)
            rows = np.maximum(joint.sum(1), 1)
            tables.append(joint)
            stats.append(dict(n=int(len(kept)), min_purity=float(pa[kept].min()) if len(kept) else None, mean_purity_a=float(pa[kept].mean()),
                              mean_purity_b=float(pur[T][kept].mean()), majority=float(joint.max(1).sum() / max(joint.sum(), 1)), nmi=ksw.norm_mi(joint.astype(float)),
                              frac_docs=float(len(kept) / keep.sum())))
        # fixed row order: greedy match rows to columns at N=1000 so the table reads along the diagonal
        ref = tables[N_GRID.index(1000)].astype(float); ref = ref / np.maximum(ref.sum(1, keepdims=True), 1)
        row_order, used = [], set()
        for col in range(k):
            cand = [(ref[r, col], r) for r in range(k) if r not in used]
            if not cand: break
            r = max(cand)[1]; row_order.append(r); used.add(r)
        row_order += [r for r in range(k) if r not in used]
        out["layers"][str(l)] = dict(row_order=[int(r) for r in row_order], group_sizes=np.bincount(ca, minlength=k).tolist(),
                                     tables=[t[row_order].tolist() for t in tables], stats=stats)
        s = stats[N_GRID.index(1000)]
        print(f"{a.tag} k={k} L{l}->L{a.target} @N=1000: min purity {s['min_purity']:.2f}, NMI {s['nmi']:.2f}, majority {s['majority']:.2f}")
    json.dump(out, open(a.out / f"{a.tag}_k{k}_grid.json", "w"))


if __name__ == "__main__":
    main()
