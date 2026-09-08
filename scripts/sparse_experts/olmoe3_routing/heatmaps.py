#!/usr/bin/env python3
"""Per-layer co-activation heatmaps for an extract_routing.py condition, in the style of the
sparse_experts co-activation report (same helpers: lift_matrix, conditional_matrix, spectral_labels,
save_png from scripts/sparse_experts/coactivation/analyze_coactivation.py):

  <tag>_lift_tok_grid.png / <tag>_lift_doc_grid.png   log2 lift, experts ordered by spectral cluster (k=8) then usage
  <tag>_cond_tok_grid.png / <tag>_cond_doc_grid.png   conditional co-activation P(j | i), same ordering
  <tag>_usage.png, <tag>_lift_hist.png               per-expert usage by layer; pairwise lift distribution

Usage: python scripts/sparse_experts/olmoe3_routing/heatmaps.py <cond_dir> --tag <name> --out <figs dir>
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location("coact", Path(__file__).resolve().parents[1] / "coactivation" / "analyze_coactivation.py")
coact = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_spec and coact)
LAYERS = list(range(1, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", type=Path); ap.add_argument("--tag", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = np.load(a.cond / "counts.npz"); Ct, Cd, N = c["coact_tok"].astype(float), c["coact_doc"].astype(float), float(c["n_tok"])
    meta = json.load(open(a.cond / "meta.json")); E = Ct.shape[1]; L = len(LAYERS); k_top = meta["top_k"]
    Nd = float(np.load(a.cond / "doc_stats.npz")["doc_len"].size)
    a.out.mkdir(parents=True, exist_ok=True)
    # cluster ordering per layer from the token lift graph (as in the earlier analysis)
    order_by_layer, Qs = [], []
    for li in range(L):
        lift, _ = coact.lift_matrix(Ct[li], N); W = np.log(np.maximum(lift, 1e-9)); W[W < 0] = 0; np.fill_diagonal(W, 0)
        lab = coact.spectral_labels(W, a.k); Qs.append(coact.modularity(W, lab))
        usage = np.diag(Ct[li]); order = np.lexsort((-usage, lab)); order_by_layer.append((order, lab))
    for name, C, NN in [("tok", Ct, N), ("doc", Cd, Nd)]:
        ncol, nrow = 3, 3
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol + 1, 4.4 * nrow), constrained_layout=True)
        for li, l in enumerate(LAYERS):
            ax = axes.flat[li]; lift, _ = coact.lift_matrix(C[li], NN); order, lab = order_by_layer[li]
            M = np.log2(np.maximum(lift[np.ix_(order, order)], 1e-3))
            im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
            for x in np.cumsum(np.bincount(lab[order], minlength=lab.max() + 1))[:-1]:
                ax.axhline(x - 0.5, color="k", lw=0.3); ax.axvline(x - 0.5, color="k", lw=0.3)
            ax.set_title(f"layer {l}  Q={Qs[li]:.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{a.tag}: log2 lift of {'token' if name == 'tok' else 'document'}-level co-activation (experts ordered by spectral cluster, then usage; clipped to ±3)", fontsize=12)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3, label="log2 lift", location="right")
        coact.save_png(fig, a.out / f"{a.tag}_lift_{name}_grid.png", 68, colors=128); plt.close(fig)
    for name, C, NN in [("tok", Ct, N), ("doc", Cd, Nd)]:
        fig, axes = plt.subplots(3, 3, figsize=(4.2 * 3 + 1, 4.4 * 3), constrained_layout=True); vmax = 0.2 if name == "tok" else 1.0
        for li, l in enumerate(LAYERS):
            ax = axes.flat[li]; P = coact.conditional_matrix(C[li]); order, lab = order_by_layer[li]
            im = ax.imshow(P[np.ix_(order, order)], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            for x in np.cumsum(np.bincount(lab[order], minlength=lab.max() + 1))[:-1]:
                ax.axhline(x - 0.5, color="w", lw=0.3, alpha=0.6); ax.axvline(x - 0.5, color="w", lw=0.3, alpha=0.6)
            offd = P[~np.eye(len(P), dtype=bool)]
            ax.set_title(f"layer {l}  max={offd.max():.2f}  p99.9={np.percentile(offd, 99.9):.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        base = (k_top - 1) / E if name == "tok" else None
        fig.suptitle(f"{a.tag}: conditional co-activation P(E_j | E_i) = N(E_i,E_j) / N(E_i) at {'token' if name == 'tok' else 'document'} level (row = conditioning expert i; same ordering as the lift grids; scale 0..{vmax:g}" + (f"; independent routing gives ~{base:.3f})" if base else ")"), fontsize=12)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3, label="P(j | i)", location="right")
        coact.save_png(fig, a.out / f"{a.tag}_cond_{name}_grid.png", 68, colors=128); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for li, l in enumerate(LAYERS):
        ax.plot(np.sort(np.diag(Ct[li]))[::-1] / N, lw=1, color=plt.cm.viridis(li / (L - 1)), label=f"L{l}" if l in (1, 5, 9) else None)
    ax.axhline(k_top / E, color="k", ls="--", lw=0.8, label=f"uniform ({k_top}/{E})"); ax.set_yscale("log"); ax.set_xlabel("expert rank"); ax.set_ylabel("token share")
    ax.set_title(f"{a.tag}: per-expert token usage by layer (dark=early, light=late)"); ax.legend(); fig.tight_layout(); coact.save_png(fig, a.out / f"{a.tag}_usage.png", 100); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for li, l in zip((0, 3, 6, 8), (1, 4, 7, 9)):
        lift, _ = coact.lift_matrix(Ct[li], N); iu = np.triu_indices(E, 1); v = lift[iu]; v = v[Ct[li][iu] > 0]
        ax.hist(np.log2(np.maximum(v, 1e-3)), bins=120, histtype="step", density=True, label=f"layer {l}")
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("log2 lift (pairs with >0 co-activations)"); ax.set_ylabel("density"); ax.set_title(f"{a.tag}: distribution of pairwise lift"); ax.legend()
    fig.tight_layout(); coact.save_png(fig, a.out / f"{a.tag}_lift_hist.png", 100); plt.close(fig)
    json.dump({"Q_spectral": Qs, "E": E, "top_k": k_top, "n_tok": N, "n_doc": Nd}, open(a.out / f"{a.tag}_heatmaps.json", "w"))
    print(f"{a.tag}: E={E} Q by layer " + " ".join(f"{q:.2f}" for q in Qs))


if __name__ == "__main__":
    main()
