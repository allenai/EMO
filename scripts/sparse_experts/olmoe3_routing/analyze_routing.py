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
        for P in POOLS:
            share = (du[:, li] * (rank < P)).sum(1) / tot
            r[f"poolable_top{P}"] = float(np.average(share, weights=w))
        d_ent = entropy(du[:, li].astype(float)); r["doc_eff_experts"] = float(np.average(np.exp(d_ent), weights=w))
        r["doc_experts_used"] = float(np.average((du[:, li] > 0).sum(1), weights=w))
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
        # Jaccard of expert sets within vs across clusters (sampled pairs)
        sets = U > 0
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
        for key in ("Q_spectral", "Q_louvain", "poolable_top64", "poolable_top128", "doc_experts_used", "mean_router_entropy"):
            lines.append(f"\n{key}")
            for cond, r in conds.items():
                lines.append(f"{cond} | {r['meta']['mean_ce']:.3f} | " + " | ".join(f"{r['layers'][str(l)][key]:.3f}" for l in LAYERS))
        lines.append("\nearly-pool conditioning: nmi(cluster;expert) / jaccard within / across")
        for cond, r in conds.items():
            lines.append(cond + " | " + " | ".join(f"{r['earlypool']['layers'][str(l)]['nmi_cluster_expert']:.3f}/{r['earlypool']['layers'][str(l)]['jaccard_within']:.2f}/{r['earlypool']['layers'][str(l)]['jaccard_across']:.2f}" for l in LAYERS))
    (a.out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
