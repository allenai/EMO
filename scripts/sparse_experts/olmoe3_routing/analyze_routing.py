#!/usr/bin/env python3
"""Analysis of extract_routing.py outputs across restriction conditions.

Input layout: <root>/<model>/<cond>/ (merged; cond = 'none' or 'A-B:P'). For every condition and
MoE layer it computes, and writes <out>/metrics.json + figures:

  usage      expert-usage entropy / effective #experts; CE of the pass (overall and per source group)
  coact      token-level lift matrix -> spectral (k=8) and Louvain modularity Q, Q of shuffled labels
  poolable   per-document share of routed assignments captured by the document's own top-P experts
             (P = 32/64/128/256, ranked by the doc's summed router scores in THAT layer); the doc's
             effective #experts; averaged over docs weighted by tokens
  crossMI    normalized mutual information between the expert at layer l and the expert at layer m
             (from the (E x E) cross-layer assignment counts), i.e. how predictable late routing is
             from early routing
  earlypool  documents clustered (k-means, k=16) by their layer-1 top-P expert SET; for later layers:
             MI(doc cluster ; expert usage) and mean Jaccard of the docs' expert sets within vs
             across clusters -> "do later layers become modular conditioned on the early pool?"

Usage: python scripts/sparse_experts/olmoe3_routing/analyze_routing.py --root <root> --out <dir>
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coactivation"))
_spec = importlib.util.spec_from_file_location("coact", Path(__file__).resolve().parents[1] / "coactivation" / "analyze_coactivation.py")
coact = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(coact)  # lift_matrix, modularity, spectral_labels, louvain_q

POOLS = (32, 64, 128, 256)
LAYERS = list(range(1, 10))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def entropy(p, axis=-1):
    p = p / np.maximum(p.sum(axis, keepdims=True), 1e-30)
    return -(p * np.log(np.maximum(p, 1e-30))).sum(axis)


def norm_mi(joint):
    """Normalized MI of a joint count matrix (rows: X, cols: Y): MI / sqrt(H(X) H(Y))."""
    P = joint / max(joint.sum(), 1e-30); px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(P * np.log(np.where(P > 0, P / (px @ py), 1)))
    hx, hy = entropy(px.ravel()), entropy(py.ravel())
    return float(mi / max(np.sqrt(hx * hy), 1e-30)), float(mi)


def kmeans_labels(X, k, seed=0):
    try:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        return kmeans2(X.astype(float), k, seed=seed, minit="++")[1]


def analyze_condition(cdir: Path, k_clusters: int, early_layer: int, early_pool: int):
    meta = json.load(open(cdir / "meta.json")); L = len(LAYERS)
    c = np.load(cdir / "counts.npz"); usage, ctok = c["usage"], c["coact_tok"]; n_tok = int(c["n_tok"])
    cross = np.load(cdir / "cross.npy", mmap_mode="r")
    du = np.load(cdir / "doc_usage.npy"); ds = np.load(cdir / "doc_scores.npy")
    st = np.load(cdir / "doc_stats.npz"); docs = np.load(cdir / "docs.npz", allow_pickle=True)
    doc_len = st["doc_len"]; ce = st["ce_sum"] / np.maximum(st["ce_len"], 1)
    stats = st["stats_sum"] / np.maximum(doc_len, 1)[:, None, None]  # (n_doc, L, 3) mean entropy/mass/top1
    group = docs["group"][docs["doc_inst"]] if "group" in docs.files else None
    res = {"meta": {k: meta[k] for k in ("restrict", "layers", "pool", "n_instances", "n_tokens", "n_docs", "mean_ce")}, "layers": {}}
    w = doc_len / doc_len.sum()
    res["ce_by_group"] = {g: float(np.average(ce[group == g], weights=doc_len[group == g])) for g in np.unique(group)} if group is not None else {}
    E = usage.shape[1]
    for li, l in enumerate(LAYERS):
        u = usage[li].astype(float); pu = u / u.sum()
        r = {"usage_entropy": float(entropy(pu)), "eff_experts": float(np.exp(entropy(pu))), "unused_experts": int((u == 0).sum()),
             "mean_router_entropy": float(np.average(stats[:, li, 0], weights=w)), "mean_top16_mass": float(np.average(stats[:, li, 1], weights=w)),
             "mean_top1_prob": float(np.average(stats[:, li, 2], weights=w))}
        # co-activation modularity
        C = ctok[li].astype(float); lift, _ = coact.lift_matrix(C, n_tok)
        W = np.log(np.maximum(lift, 1e-9)); W[W < 0] = 0; np.fill_diagonal(W, 0)
        lab = coact.spectral_labels(W, k_clusters); q = coact.modularity(W, lab)
        rng = np.random.default_rng(0); q_null = float(np.mean([coact.modularity(W, rng.permutation(lab)) for _ in range(5)]))
        ql, labl = coact.louvain_q(W)
        r.update(Q_spectral=float(q), Q_null=q_null, Q_louvain=float(ql), n_louvain=int(len(np.unique(labl))))
        # poolability: share of the doc's assignments inside its own top-P experts (by doc scores)
        order = np.argsort(-ds[:, li], axis=1)  # (n_doc, E) experts ranked by doc score
        rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(E)[None, :].repeat(len(order), 0), axis=1)
        tot = np.maximum(du[:, li].sum(1), 1)
        okdoc = doc_len >= 64  # unweighted stats over docs with >= 64 tokens
        for P in POOLS:
            share = (du[:, li] * (rank < P)).sum(1) / tot
            r[f"poolable_top{P}"] = float(np.average(share, weights=w))
            r[f"poolable_top{P}_unw"] = float(share[okdoc].mean())
        d_ent = entropy(du[:, li].astype(float)); r["doc_eff_experts"] = float(np.average(np.exp(d_ent), weights=w))
        r["doc_eff_experts_unw"] = float(np.exp(d_ent)[okdoc].mean())
        r["doc_experts_used"] = float(np.average((du[:, li] > 0).sum(1), weights=w))
        r["doc_experts_used_unw"] = float((du[okdoc, li] > 0).sum(1).mean())
        res["layers"][str(l)] = r
    # cross-layer normalized MI (expert at l vs expert at m)
    nmi = np.zeros((L, L))
    for li in range(L):
        for mi_ in range(L):
            nmi[li, mi_] = norm_mi(np.asarray(cross[li, mi_], dtype=float))[0]
    res["cross_nmi"] = nmi.tolist()
    # early-pool conditioning: cluster docs by their early-layer top-P set, evaluate later layers
    el = LAYERS.index(early_layer)
    order = np.argsort(-ds[:, el], axis=1)[:, :early_pool]
    X = np.zeros((len(ds), E), np.float32); np.put_along_axis(X, order, 1.0, axis=1)
    keep = doc_len >= 64  # ignore tiny docs
    lab = kmeans_labels(X[keep], k_clusters)
    ep = {}
    rng = np.random.default_rng(0)
    for li, l in enumerate(LAYERS):
        U = du[keep, li].astype(float)
        joint = np.stack([U[lab == c].sum(0) for c in range(k_clusters)])  # (k, E)
        nmi_c, _ = norm_mi(joint)
        # Jaccard of each doc's top-64 expert set (by usage) within vs across clusters (sampled pairs)
        top = np.argsort(-U, axis=1)[:, :64]
        sets = np.zeros(U.shape, bool); np.put_along_axis(sets, top, True, axis=1)
        def mean_j(pairs):
            a, b = sets[pairs[:, 0]], sets[pairs[:, 1]]
            inter = (a & b).sum(1); union = (a | b).sum(1)
            return float(np.mean(inter / np.maximum(union, 1)))
        n = len(U); i = rng.integers(0, n, 20000); j = rng.integers(0, n, 20000); m = i != j
        same = lab[i] == lab[j]
        ep[str(l)] = {"nmi_cluster_expert": nmi_c, "jaccard_within": mean_j(np.stack([i[m & same], j[m & same]], 1)),
                      "jaccard_across": mean_j(np.stack([i[m & ~same], j[m & ~same]], 1))}
    res["earlypool"] = {"early_layer": early_layer, "early_pool": early_pool, "k": k_clusters, "n_docs": int(keep.sum()), "layers": ep}
    return res


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def cond_style(cond):
    """colour by pool size, line style by restricted prefix."""
    if cond == "none":
        return dict(color="black", ls="-", lw=2.2, label="unrestricted")
    pre, P = cond.split(":"); P = int(P)
    colors = {32: "#d62728", 64: "#ff7f0e", 128: "#2ca02c", 256: "#1f77b4"}
    ls = {"1-3": "-", "1-6": "--", "1-9": ":"}[pre]
    return dict(color=colors[P], ls=ls, lw=1.6, label=f"layers {pre} @ pool {P}")


def make_figures(all_res, out: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figs"; figs.mkdir(parents=True, exist_ok=True)
    models = list(all_res)
    def per_layer(model, cond, key, sub=None):
        r = all_res[model][cond]
        return [r[sub]["layers"][str(l)][key] if sub else r["layers"][str(l)][key] for l in LAYERS]
    def line_fig(key, ylabel, fname, sub=None, transform=None):
        fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models):
            for cond in sorted(all_res[model], key=lambda c: (c != "none", c.split(":")[0], int(c.split(":")[1]) if ":" in c else 0)):
                y = transform(all_res[model][cond]) if transform else per_layer(model, cond, key, sub)
                ax.plot(LAYERS, y, marker="o", ms=3, **cond_style(cond))
            ax.set_title(f"{model}"); ax.set_xlabel("MoE layer"); ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel); axes[-1].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(figs / fname, dpi=110); plt.close(fig)
    line_fig("poolable_top64_unw", "share of routed assignments inside the doc's top-64 experts", "poolable64_by_layer.png")
    line_fig("poolable_top128_unw", "share inside the doc's top-128 experts", "poolable128_by_layer.png")
    line_fig("doc_eff_experts_unw", "effective # experts per document", "doc_eff_experts_by_layer.png")
    line_fig("Q_louvain", "Louvain modularity Q (token co-activation lift)", "q_louvain_by_layer.png")
    line_fig("mean_router_entropy", "mean router entropy (nats)", "router_entropy_by_layer.png")
    line_fig(None, "Jaccard(top-64 sets) within - across early-pool clusters", "earlypool_jaccard_gap.png",
             transform=lambda r: [r["earlypool"]["layers"][str(l)]["jaccard_within"] - r["earlypool"]["layers"][str(l)]["jaccard_across"] for l in LAYERS])
    line_fig(None, "NMI(early-pool cluster ; expert usage)", "earlypool_nmi.png",
             transform=lambda r: [r["earlypool"]["layers"][str(l)]["nmi_cluster_expert"] for l in LAYERS])
    # cross-layer NMI heatmaps: unrestricted vs the tightest 1-3 and 1-6 restrictions
    show = ["none", "1-3:32", "1-6:32", "1-9:32"]
    fig, axes = plt.subplots(len(models), len(show), figsize=(3.3 * len(show), 3.2 * len(models)), squeeze=False)
    vmax = max(np.max(np.array(all_res[m][c]["cross_nmi"])[~np.eye(len(LAYERS), dtype=bool)]) for m in models for c in show if c in all_res[m])
    for i, model in enumerate(models):
        for j, cond in enumerate(show):
            ax = axes[i, j]
            if cond not in all_res[model]: ax.axis("off"); continue
            M = np.array(all_res[model][cond]["cross_nmi"]); np.fill_diagonal(M, np.nan)
            im = ax.imshow(M, vmin=0, vmax=vmax, cmap="viridis"); ax.set_title(f"{model} / {cond}", fontsize=9)
            ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels(LAYERS, fontsize=7); ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(LAYERS, fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="normalized MI(expert@l ; expert@m)")
    fig.savefig(figs / "cross_layer_nmi.png", dpi=110); plt.close(fig)
    # CE by condition
    fig, ax = plt.subplots(figsize=(10, 3.6))
    conds = sorted({c for m in models for c in all_res[m]}, key=lambda c: (c != "none", c.split(":")[0], int(c.split(":")[1]) if ":" in c else 0))
    x = np.arange(len(conds)); wdt = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + i * wdt, [all_res[m][c]["meta"]["mean_ce"] if c in all_res[m] else np.nan for c in conds], wdt, label=m)
    ax.set_xticks(x + wdt * (len(models) - 1) / 2); ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8); ax.set_ylabel("mean CE"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    lo = min(all_res[m][c]["meta"]["mean_ce"] for m in models for c in all_res[m]); ax.set_ylim(lo - 0.05, None)
    fig.tight_layout(); fig.savefig(figs / "ce_by_condition.png", dpi=110); plt.close(fig)
    log(f"figures -> {figs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-clusters", type=int, default=8)
    ap.add_argument("--early-layer", type=int, default=1)
    ap.add_argument("--early-pool", type=int, default=32)
    ap.add_argument("--models", default="std,emo")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for model in a.models.split(","):
        for cdir in sorted((a.root / model).glob("*")):
            if not (cdir / "DONE").exists() or not (cdir / "counts.npz").exists():
                continue
            log(f"{model}/{cdir.name}"); all_res.setdefault(model, {})[cdir.name] = analyze_condition(cdir, a.k_clusters, a.early_layer, a.early_pool)
    json.dump(all_res, open(a.out / "metrics.json", "w"), indent=1)
    # compact tables
    lines = []
    for model, conds in all_res.items():
        lines.append(f"\n## {model}\ncond | CE | " + " | ".join(f"L{l}" for l in LAYERS))
        for key in ("Q_spectral", "Q_louvain", "poolable_top64", "poolable_top64_unw", "poolable_top128_unw", "doc_eff_experts_unw", "mean_router_entropy"):
            lines.append(f"\n{key}")
            for cond, r in conds.items():
                lines.append(f"{cond} | {r['meta']['mean_ce']:.3f} | " + " | ".join(f"{r['layers'][str(l)][key]:.3f}" for l in LAYERS))
        lines.append("\nearly-pool conditioning: nmi(cluster;expert) / jaccard within / across")
        for cond, r in conds.items():
            lines.append(cond + " | " + " | ".join(f"{r['earlypool']['layers'][str(l)]['nmi_cluster_expert']:.3f}/{r['earlypool']['layers'][str(l)]['jaccard_within']:.2f}/{r['earlypool']['layers'][str(l)]['jaccard_across']:.2f}" for l in LAYERS))
    (a.out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))
    make_figures(all_res, a.out)


if __name__ == "__main__":
    main()
