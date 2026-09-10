#!/usr/bin/env python3
"""Spectral-cluster-count sweep + document partition from layer-1 expert clusters.

For each k in --ks, on one extract_routing.py condition:
  (a) per-layer spectral clusters (k) of the token-level log-lift graph -> <tag>_k<k>_lift_tok_grid.png
      (log2 lift, experts ordered by cluster then usage) and spectral Q vs shuffled-label null;
  (b) LAYER-1 experts partitioned into k clusters; every document is assigned to the cluster that
      captures the most of its layer-1 routed assignments (doc_usage). Reported per k:
        purity          mean over docs (>=64 tokens) of the share of layer-1 assignments inside the
                        assigned cluster; null = random expert partition with the same cluster sizes
        frac_purity>.5  fraction of documents with purity > 0.5 (and null)
        lift_within/across   mean log2 lift between the experts a document actually uses at layer 1
                        (its top-16 by usage) that fall in its assigned cluster vs outside it
        doc-cluster sizes and source composition (NMI(doc cluster ; source group))
Usage: python k_sweep_partition.py <cond_dir> --tag emo1000_full --out <dir> [--ks 4,8,16,32,64]
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location("coact", Path(__file__).resolve().parents[1] / "coactivation" / "analyze_coactivation.py")
coact = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(coact)
LAYERS = list(range(1, 10))


def norm_mi(joint):
    P = joint / max(joint.sum(), 1e-30); px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(P * np.log(np.where(P > 0, P / (px @ py), 1)))
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(mi / max(np.sqrt(h(px.ravel()) * h(py.ravel())), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", type=Path); ap.add_argument("--tag", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ks", default="4,8,16,32,64"); ap.add_argument("--min-doc-tokens", type=int, default=64)
    ap.add_argument("--layer", type=int, default=1, help="MoE layer whose expert partition defines the document partition")
    ap.add_argument("--no-grids", action="store_true")
    a = ap.parse_args(); KS = [int(k) for k in a.ks.split(",")]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = np.load(a.cond / "counts.npz"); Ct = c["coact_tok"].astype(float); N = float(c["n_tok"])
    du = np.load(a.cond / "doc_usage.npy"); st = np.load(a.cond / "doc_stats.npz"); doc_len = st["doc_len"]
    docs = np.load(a.cond / "docs.npz", allow_pickle=True); group = docs["group"][docs["doc_inst"]]
    E = Ct.shape[1]; L = len(LAYERS); rng = np.random.default_rng(0); a.out.mkdir(parents=True, exist_ok=True)
    keep = doc_len >= a.min_doc_tokens; PL = LAYERS.index(a.layer); U1 = du[keep, PL].astype(float); grp = group[keep]; groups = sorted(set(grp))
    lifts, Ws = [], []
    for li in range(L):
        lift, _ = coact.lift_matrix(Ct[li], N); W = np.log(np.maximum(lift, 1e-9)); W[W < 0] = 0; np.fill_diagonal(W, 0); lifts.append(lift); Ws.append(W)
    log2lift1 = np.log2(np.maximum(lifts[PL], 1e-3))
    results = {}
    for k in KS:
        labs, Qs, Qn = [], [], []
        for li in range(L):
            lab = coact.spectral_labels(Ws[li], k); labs.append(lab); Qs.append(coact.modularity(Ws[li], lab))
            Qn.append(float(np.mean([coact.modularity(Ws[li], rng.permutation(lab)) for _ in range(3)])))
        # (a) grid
        if a.no_grids: labs_grid = None
        fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3 + 1, 4.4 * 3), constrained_layout=True)
        for li, l in enumerate(LAYERS):
            ax = axes.flat[li]; lab = labs[li]; order = np.lexsort((-np.diag(Ct[li]), lab))
            im = ax.imshow(np.log2(np.maximum(lifts[li][np.ix_(order, order)], 1e-3)), cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
            for x in np.cumsum(np.bincount(lab[order], minlength=k))[:-1]: ax.axhline(x - 0.5, color="k", lw=0.25); ax.axvline(x - 0.5, color="k", lw=0.25)
            ax.set_title(f"layer {l}  Q={Qs[li]:.2f} (null {Qn[li]:.2f})", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{a.tag}: log2 token-level lift, experts ordered by spectral cluster (k={k}) then usage", fontsize=12)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3, label="log2 lift", location="right")
        if not a.no_grids: coact.save_png(fig, a.out / f"{a.tag}_k{k}_lift_tok_grid.png", 68, colors=128)
        plt.close(fig)
        # (b) layer-1 expert partition -> document partition
        lab1 = labs[PL]; onehot = np.zeros((E, k)); onehot[np.arange(E), lab1] = 1
        mass = U1 @ onehot; tot = np.maximum(mass.sum(1), 1); dcl = mass.argmax(1); purity = mass.max(1) / tot
        null_p = []
        for _ in range(5):
            perm = rng.permutation(lab1); oh = np.zeros((E, k)); oh[np.arange(E), perm] = 1; m = U1 @ oh; null_p.append(m.max(1) / tot)
        null_p = np.stack(null_p)
        # lift within vs across, over each doc's top-16 layer-1 experts
        top = np.argsort(-U1, axis=1)[:, :16]
        within, across = [], []
        for d in range(len(U1)):
            e = top[d][U1[d, top[d]] > 0]; if_in = lab1[e] == dcl[d]
            ein, eout = e[if_in], e[~if_in]
            if len(ein) > 1: within.append(log2lift1[np.ix_(ein, ein)][np.triu_indices(len(ein), 1)].mean())
            if len(ein) and len(eout): across.append(log2lift1[np.ix_(ein, eout)].mean())
        sizes = np.bincount(dcl, minlength=k); esizes = np.bincount(lab1, minlength=k)
        joint = np.zeros((k, len(groups)))
        for gi, g in enumerate(groups): joint[:, gi] = np.bincount(dcl[grp == g], minlength=k)
        results[k] = dict(Q_spectral=Qs, Q_null=Qn, purity=float(purity.mean()), purity_null=float(null_p.mean()),
                          frac_purity_gt_half=float((purity > 0.5).mean()), frac_purity_gt_half_null=float((null_p > 0.5).mean()),
                          purity_median=float(np.median(purity)), lift_within=float(np.mean(within)), lift_across=float(np.mean(across)),
                          expert_cluster_sizes=esizes.tolist(), doc_cluster_sizes=sizes.tolist(), nmi_doc_cluster_source=norm_mi(joint),
                          source_composition={g: (joint[:, gi] / np.maximum(sizes, 1)).round(3).tolist() for gi, g in enumerate(groups)}, n_docs=int(keep.sum()))
        print(f"k={k:3d}: L{a.layer} Q={Qs[PL]:.3f} (null {Qn[PL]:.3f}) | doc purity {purity.mean():.3f} (null {null_p.mean():.3f}), median {np.median(purity):.3f}, >0.5: {(purity>0.5).mean():.2f} (null {(null_p>0.5).mean():.2f}) | log2 lift within {np.mean(within):.2f} vs across {np.mean(across):.2f} | expert cluster sizes {sorted(esizes.tolist(), reverse=True)[:6]}.. | doc cluster sizes {sorted(sizes.tolist(), reverse=True)[:6]}.. | NMI(doc cluster;source)={results[k]['nmi_doc_cluster_source']:.3f}")
        # purity histogram (source composition is kept in the json only)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.hist(purity, bins=50, range=(0, 1), alpha=0.7, label=f"layer-{a.layer} expert clusters"); ax.hist(null_p[0], bins=50, range=(0, 1), alpha=0.5, label="random partition (same sizes)")
        ax.set_xlabel(f"document purity: share of layer-{a.layer} assignments in its assigned cluster"); ax.legend(fontsize=8); ax.set_title(f"k={k}")
        fig.tight_layout(); coact.save_png(fig, a.out / f"{a.tag}_L{a.layer}_k{k}_docpartition.png", 100); plt.close(fig)
    # Q vs k summary figure
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for li, l in enumerate(LAYERS): ax.plot(KS, [results[k]["Q_spectral"][li] for k in KS], marker="o", ms=3, color=plt.cm.viridis(li / (L - 1)), label=f"L{l}" if l in (1, 5, 9) else None)
    ax.plot(KS, [results[k]["Q_null"][0] for k in KS], "k--", lw=0.8, label="shuffled null (L1)"); ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_xlabel("k (spectral clusters)"); ax.set_ylabel("modularity Q"); ax.legend(); ax.grid(alpha=0.3); ax.set_title(f"{a.tag}: spectral Q vs k (dark=early layer, light=late)")
    fig.tight_layout(); coact.save_png(fig, a.out / f"{a.tag}_Q_vs_k.png", 100); plt.close(fig)
    json.dump({str(k): v for k, v in results.items()}, open(a.out / f"{a.tag}_L{a.layer}_ksweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
