#!/usr/bin/env python3
"""
Analysis of expert co-activation counts produced by extract_coactivation.py.

For each routing pass (`pool_config` = full routing, `pool_64` = eval pool pinned to 64) and each
layer this computes, from the per-source count tensors C_tok / C_doc (src, L, E, E):

  usage           p_i = C_ii / N; effective #experts exp(H(p)); Gini
  lift            C_ij * N / (C_i * C_j)  (observed / expected-under-independence); PMI = log lift
  block structure spectral clustering of the token-level lift graph (k clusters), Newman
                  modularity Q of that partition on the co-activation graph, Q of a shuffled
                  partition as the null; Louvain Q when networkx is available
  token vs doc    Spearman rho between token-level and doc-level lift (pairs with support)
  per source      the same usage / Q per source; Jaccard overlap of the top-1% pairs (by lift)
                  between sources; per-source usage vectors cosine similarity
  pool 64 vs full usage shift, Q, unique experts per doc (from raw top-k)

Aggregates: "sample" (counts as sampled) and "natural" (per-source counts reweighted to the
window's token shares, i.e. the training distribution). Writes analysis.json + PNG figures into
`--out` (default claude_outputs/sparse_experts/coactivation/<run>/). Shared expert (last index)
is dropped from every matrix before analysis.

  PYTHONPATH=src python scripts/sparse_experts/coactivation/analyze_coactivation.py \
      --run-dir sparse_experts/coactivation/sparse_8of512_10b_step2384
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# token shares of the 20B-40B window (from the cluster-label file over all 18.3M docs)
NATURAL_TOKEN_SHARE = {
    "dclm": 0.934,
    "proof-pile-2": 0.032,
    "starcoder": 0.018,
    "pes2o": 0.016,
    "olmo-mix": 0.001,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------
def usage_stats(diag: np.ndarray):
    p = diag / max(diag.sum(), 1)
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    srt = np.sort(p)
    n = len(p)
    gini = (
        float((2 * np.arange(1, n + 1) - n - 1) @ srt / (n * srt.sum())) if srt.sum() > 0 else 0.0
    )
    return dict(
        effective_experts=float(np.exp(H)),
        entropy=H,
        gini=gini,
        unused=int((diag == 0).sum()),
        max_share=float(p.max()),
        min_share=float(p.min()),
    )


def lift_matrix(C: np.ndarray, N: float):
    """C: (E, E) symmetric co-activation counts (diag = usage). Returns lift, expected."""
    c = np.diag(C).astype(np.float64)
    expected = np.outer(c, c) / max(N, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(expected > 0, C / expected, 0.0)
    np.fill_diagonal(lift, 0.0)
    return lift, expected


def conditional_matrix(C: np.ndarray) -> np.ndarray:
    """P(j active | i active) = C_ij / C_ii, rows = conditioning expert i; zero diagonal."""
    c = np.diag(C).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(c[:, None] > 0, C / c[:, None], 0.0)
    np.fill_diagonal(P, 0.0)
    return P


def modularity(W: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of partition `labels` on weighted undirected graph W (zero diagonal)."""
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    m2 = W.sum()
    if m2 <= 0:
        return 0.0
    k = W.sum(1)
    Q = 0.0
    for c in np.unique(labels):
        idx = labels == c
        Q += W[np.ix_(idx, idx)].sum() / m2 - (k[idx].sum() / m2) ** 2
    return float(Q)


def spectral_labels(A: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Normalized spectral clustering (Ng-Jordan-Weiss) on affinity A >= 0 (zero diagonal)."""
    from scipy.linalg import eigh

    A = np.maximum(A, 0.0)
    d = A.sum(1)
    d_is = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Lsym = np.eye(len(A)) - (d_is[:, None] * A * d_is[None, :])
    vals, vecs = eigh(Lsym, subset_by_index=[0, k - 1])
    X = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    try:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    except ImportError:
        from scipy.cluster.vq import kmeans2

        _, lab = kmeans2(X, k, minit="++", seed=seed)
        return lab


def louvain_q(W: np.ndarray, seed: int = 0):
    try:
        import networkx as nx
    except ImportError:
        return None, None
    G = nx.Graph()
    E = len(W)
    G.add_nodes_from(range(E))
    ii, jj = np.nonzero(np.triu(W, 1))
    G.add_weighted_edges_from((int(i), int(j), float(W[i, j])) for i, j in zip(ii, jj))
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    lab = np.zeros(E, dtype=int)
    for c, members in enumerate(comms):
        lab[list(members)] = c
    return float(nx.community.modularity(G, comms, weight="weight")), lab


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0


def top_pairs_set(lift: np.ndarray, support: np.ndarray, frac: float, min_support: int):
    iu = np.triu_indices(len(lift), 1)
    ok = support[iu] >= min_support
    vals = np.where(ok, lift[iu], -np.inf)
    n = max(1, int(frac * ok.sum()))
    top = np.argpartition(-vals, n - 1)[:n]
    return set(zip(iu[0][top].tolist(), iu[1][top].tolist()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


# --------------------------------------------------------------------------------------------
# per-pool analysis
# --------------------------------------------------------------------------------------------
def analyze_pool(pool_dir: Path, sources: list, k_clusters: int, figs: Path, tag: str, plot):
    z = np.load(pool_dir / "counts.npz")
    c_tok_src, c_doc_src, n_tok_src, n_doc_src = z["c_tok"], z["c_doc"], z["n_tok"], z["n_doc"]
    S, L, E, _ = c_tok_src.shape
    Es = E - 1  # drop shared expert (last index)
    c_tok_src = c_tok_src[:, :, :Es, :Es].astype(np.float64)
    c_doc_src = c_doc_src[:, :, :Es, :Es].astype(np.float64)
    info = json.load(open(pool_dir / "info.json"))

    # aggregates
    w_nat = np.array([NATURAL_TOKEN_SHARE[s] for s in sources])
    w_nat = w_nat / w_nat.sum()
    sample_share = n_tok_src / n_tok_src.sum()
    reweight = w_nat / np.maximum(sample_share, 1e-12)
    aggs = {
        "sample": (
            c_tok_src.sum(0),
            c_doc_src.sum(0),
            float(n_tok_src.sum()),
            float(n_doc_src.sum()),
        ),
        "natural": (
            np.tensordot(reweight, c_tok_src, 1),
            np.tensordot(reweight, c_doc_src, 1),
            float((reweight * n_tok_src).sum()),
            float((reweight * n_doc_src).sum()),
        ),
    }
    res = dict(
        pool=tag,
        num_tokens=int(n_tok_src.sum()),
        num_docs=int(n_doc_src.sum()),
        n_tok_per_source=dict(zip(sources, n_tok_src.tolist())),
        n_doc_per_source=dict(zip(sources, n_doc_src.tolist())),
        natural_reweight=dict(zip(sources, reweight.tolist())),
        layers={},
        per_source={},
        eval_document_expert_pool=info["ranks"][0]["eval_document_expert_pool"],
    )

    rng = np.random.default_rng(0)
    order_by_layer = {}
    for agg_name, (Ct, Cd, N, Nd) in aggs.items():
        for li in range(L):
            lift_t, exp_t = lift_matrix(Ct[li], N)
            lift_d, _ = lift_matrix(Cd[li], Nd)
            diag = np.diag(Ct[li])
            W = Ct[li].copy()
            np.fill_diagonal(W, 0.0)
            r = dict(usage=usage_stats(diag))
            iu = np.triu_indices(Es, 1)
            support = Ct[li][iu]
            r["pairs"] = dict(
                never_coactive_frac=float((support == 0).mean()),
                lift_median=float(np.median(lift_t[iu][support > 0]))
                if (support > 0).any()
                else 0.0,
                lift_p90=float(np.percentile(lift_t[iu][support > 0], 90))
                if (support > 0).any()
                else 0.0,
                lift_p99=float(np.percentile(lift_t[iu][support > 0], 99))
                if (support > 0).any()
                else 0.0,
                lift_max=float(lift_t.max()),
                frac_pairs_lift_gt2=float((lift_t[iu] > 2).mean()),
                frac_pairs_lift_gt5=float((lift_t[iu] > 5).mean()),
            )
            if agg_name == "sample":
                lab = spectral_labels(lift_t, k_clusters)
                q_spec = modularity(W, lab)
                q_null = float(np.mean([modularity(W, rng.permutation(lab)) for _ in range(5)]))
                q_louv, lab_louv = louvain_q(W)
                r["structure"] = dict(
                    k=k_clusters,
                    Q_spectral=q_spec,
                    Q_null_shuffled=q_null,
                    Q_louvain=q_louv,
                    n_louvain_communities=int(len(np.unique(lab_louv)))
                    if lab_louv is not None
                    else None,
                    cluster_sizes=np.bincount(lab, minlength=k_clusters).tolist(),
                )
                # within-cluster share of co-activation mass
                within = sum(W[np.ix_(lab == c, lab == c)].sum() for c in range(k_clusters)) / max(
                    W.sum(), 1
                )
                r["structure"]["within_cluster_mass"] = float(within)
                # ordering for heatmaps: clusters, then usage within cluster
                order = np.lexsort((-diag, lab))
                order_by_layer[li] = (order, lab)
                mask = support > 0
                r["token_vs_doc_spearman"] = (
                    spearman(lift_t[iu][mask], lift_d[iu][mask]) if mask.sum() > 10 else 0.0
                )
                # conditional co-activation P(j | i) = C_ij / C_i (asymmetric)
                cond = conditional_matrix(Ct[li])
                offd = cond[~np.eye(Es, dtype=bool)]
                ii, jj = np.unravel_index(np.argsort(-cond, axis=None)[:200], cond.shape)
                topc = [
                    dict(
                        i=int(a),
                        j=int(b),
                        p_j_given_i=float(cond[a, b]),
                        count=int(Ct[li][a, b]),
                        n_i=int(diag[a]),
                    )
                    for a, b in zip(ii, jj)
                    if a != b and diag[a] >= 1000
                ][:10]
                r["conditional"] = dict(
                    max=float(offd.max()),
                    p999=float(np.percentile(offd, 99.9)),
                    median=float(np.median(offd)),
                    independence_baseline=float(7.0 / Es),
                    frac_pairs_gt_0p1=float((offd > 0.1).mean()),
                    frac_pairs_gt_0p25=float((offd > 0.25).mean()),
                    top=topc,
                )
                # top pairs
                top = top_pairs_set(lift_t, Ct[li], 0.0002, 50)
                r["top_pairs"] = sorted(
                    [
                        dict(i=int(i), j=int(j), lift=float(lift_t[i, j]), count=int(Ct[li][i, j]))
                        for i, j in top
                    ],
                    key=lambda x: -x["lift"],
                )[:15]
            res["layers"].setdefault(str(li), {})[agg_name] = r

    # per-source
    for si, s in enumerate(sources):
        ps = dict(layers={})
        for li in range(L):
            Ct = c_tok_src[si, li]
            N = float(n_tok_src[si])
            lift_t, _ = lift_matrix(Ct, N)
            W = Ct.copy()
            np.fill_diagonal(W, 0.0)
            lab = spectral_labels(lift_t, k_clusters)
            ps["layers"][str(li)] = dict(
                usage=usage_stats(np.diag(Ct)), Q_spectral=modularity(W, lab)
            )
        res["per_source"][s] = ps
    # cross-source agreement per layer: usage cosine + top-1% pair overlap
    xs = {}
    for li in range(L):
        u = np.stack([np.diag(c_tok_src[si, li]) for si in range(S)])
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        cos = (u @ u.T).tolist()
        tops = [
            top_pairs_set(
                lift_matrix(c_tok_src[si, li], float(n_tok_src[si]))[0], c_tok_src[si, li], 0.01, 20
            )
            for si in range(S)
        ]
        jac = [[jaccard(tops[a], tops[b]) for b in range(S)] for a in range(S)]
        xs[str(li)] = dict(usage_cosine=cos, top1pct_pair_jaccard=jac)
    res["cross_source"] = dict(sources=sources, layers=xs)

    # unique experts per doc per layer from raw top-k (sample of docs)
    topk = np.load(pool_dir / "topk.npy", mmap_mode="r")
    tok_doc = np.load(pool_dir / "tok_doc.npy")
    docs = np.unique(tok_doc)
    sel = rng.choice(docs, size=min(2000, len(docs)), replace=False)
    order_idx = np.argsort(tok_doc, kind="stable")
    sorted_docs = tok_doc[order_idx]
    uniq = np.zeros((len(sel), L), dtype=np.int32)
    for n, d in enumerate(sel):
        lo, hi = np.searchsorted(sorted_docs, [d, d + 1])
        rows = np.sort(order_idx[lo:hi])
        t = np.asarray(topk[rows])  # (T, L, k)
        for li in range(L):
            uniq[n, li] = len(np.unique(t[:, li]))
    res["unique_experts_per_doc"] = dict(
        n_docs=int(len(sel)),
        mean_by_layer=uniq.mean(0).tolist(),
        p50_by_layer=np.percentile(uniq, 50, axis=0).tolist(),
        p90_by_layer=np.percentile(uniq, 90, axis=0).tolist(),
        max_by_layer=uniq.max(0).tolist(),
    )

    if plot:
        make_pool_figures(res, aggs["sample"], order_by_layer, Es, L, figs, tag)
    return res


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def save_png(fig, path, dpi):
    """Save then quantise to an 8-bit palette (near-lossless for heatmaps, ~3x smaller; the report
    page embeds every figure and must stay under Cloudflare Pages' 25 MiB file limit)."""
    fig.savefig(path, dpi=dpi)
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        im.save(path, optimize=True)
    except ImportError:
        pass


def make_pool_figures(res, agg, order_by_layer, Es, L, figs: Path, tag: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ct, Cd, N, Nd = agg
    figs.mkdir(parents=True, exist_ok=True)
    # heatmap grid: log2 lift, cluster-ordered, per layer
    for name, C, NN in [("tok", Ct, N), ("doc", Cd, Nd)]:
        ncol = 4
        nrow = int(np.ceil(L / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.2 * ncol + 1, 4.4 * nrow), constrained_layout=True
        )
        for li in range(L):
            ax = axes.flat[li]
            lift, _ = lift_matrix(C[li], NN)
            order, lab = order_by_layer[li]
            M = np.log2(np.maximum(lift[np.ix_(order, order)], 1e-3))
            im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
            # cluster boundaries
            b = np.cumsum(np.bincount(lab[order], minlength=lab.max() + 1))[:-1]
            for x in b:
                ax.axhline(x - 0.5, color="k", lw=0.3)
                ax.axvline(x - 0.5, color="k", lw=0.3)
            q = res["layers"][str(li)]["sample"]["structure"]["Q_spectral"]
            ax.set_title(f"layer {li}  Q={q:.2f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.flat[L:]:
            ax.axis("off")
        fig.suptitle(
            f"{tag}: log2 lift of {'token' if name == 'tok' else 'document'}-level co-activation "
            f"(experts ordered by spectral cluster, then usage; clipped to ±3)",
            fontsize=12,
        )
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3, label="log2 lift", location="right")
        save_png(fig, figs / f"{tag}_lift_{name}_grid.png", 80)
        plt.close(fig)
    # conditional co-activation grids: P(j | i) = C_ij / C_i, rows = conditioning expert
    for name, C, NN in [("tok", Ct, N), ("doc", Cd, Nd)]:
        ncol = 4
        nrow = int(np.ceil(L / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.2 * ncol + 1, 4.4 * nrow), constrained_layout=True
        )
        vmax = 0.2 if name == "tok" else 1.0
        for li in range(L):
            ax = axes.flat[li]
            P = conditional_matrix(C[li])
            order, lab = order_by_layer[li]
            im = ax.imshow(
                P[np.ix_(order, order)], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"
            )
            b = np.cumsum(np.bincount(lab[order], minlength=lab.max() + 1))[:-1]
            for x in b:
                ax.axhline(x - 0.5, color="w", lw=0.3, alpha=0.6)
                ax.axvline(x - 0.5, color="w", lw=0.3, alpha=0.6)
            offd = P[~np.eye(len(P), dtype=bool)]
            ax.set_title(
                f"layer {li}  max={offd.max():.2f}  p99.9={np.percentile(offd, 99.9):.2f}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.flat[L:]:
            ax.axis("off")
        base = 7.0 / Es if name == "tok" else None
        fig.suptitle(
            f"{tag}: conditional co-activation P(E_j | E_i) = N(E_i,E_j) / N(E_i) at "
            f"{'token' if name == 'tok' else 'document'} level (row = conditioning expert i, col = j; "
            f"same cluster ordering as the lift grids; scale 0..{vmax:g}"
            + (f"; independent routing gives ~{base:.3f})" if base else ")"),
            fontsize=12,
        )
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3, label="P(j | i)", location="right")
        save_png(fig, figs / f"{tag}_cond_{name}_grid.png", 80)
        plt.close(fig)
    # usage per layer (sorted, log)
    fig, ax = plt.subplots(figsize=(9, 4))
    for li in range(L):
        d = np.sort(np.diag(Ct[li]))[::-1] / N
        ax.plot(
            d,
            lw=1,
            color=plt.cm.viridis(li / max(L - 1, 1)),
            label=f"L{li}" if li in (0, L // 2, L - 1) else None,
        )
    ax.axhline(7 / Es, color="k", ls="--", lw=0.8, label="uniform (7/511)")
    ax.set_yscale("log")
    ax.set_xlabel("expert rank")
    ax.set_ylabel("token share")
    ax.set_title(f"{tag}: per-expert token usage by layer (dark=early, light=late)")
    ax.legend()
    fig.tight_layout()
    save_png(fig, figs / f"{tag}_usage.png", 100)
    plt.close(fig)
    # lift distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    for li in range(0, L, max(1, L // 4)):
        lift, _ = lift_matrix(Ct[li], N)
        iu = np.triu_indices(Es, 1)
        v = lift[iu]
        v = v[Ct[li][iu] > 0]
        ax.hist(
            np.log2(np.maximum(v, 1e-3)),
            bins=120,
            histtype="step",
            density=True,
            label=f"layer {li}",
        )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("log2 lift (pairs with >0 co-activations)")
    ax.set_ylabel("density")
    ax.set_title(f"{tag}: distribution of pairwise lift")
    ax.legend()
    fig.tight_layout()
    save_png(fig, figs / f"{tag}_lift_hist.png", 100)
    plt.close(fig)


def make_summary_figures(results: dict, figs: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = list(results)
    L = len(results[tags[0]]["layers"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for tag in tags:
        lay = results[tag]["layers"]
        q = [lay[str(li)]["sample"]["structure"]["Q_spectral"] for li in range(L)]
        qn = [lay[str(li)]["sample"]["structure"]["Q_null_shuffled"] for li in range(L)]
        ql = [lay[str(li)]["sample"]["structure"]["Q_louvain"] for li in range(L)]
        axes[0].plot(q, marker="o", label=f"{tag} spectral")
        if all(x is not None for x in ql):
            axes[0].plot(ql, marker="s", ls="--", label=f"{tag} louvain")
        axes[0].plot(
            qn, ls=":", color="gray", label=f"{tag} shuffled null" if tag == tags[0] else None
        )
        eff = [lay[str(li)]["sample"]["usage"]["effective_experts"] for li in range(L)]
        axes[1].plot(eff, marker="o", label=tag)
        sp = [lay[str(li)]["sample"].get("token_vs_doc_spearman", 0) for li in range(L)]
        axes[2].plot(sp, marker="o", label=tag)
    axes[0].set_title("modularity Q of token co-activation graph")
    axes[1].set_title("effective #experts exp(H(usage)) (of 511)")
    axes[2].set_title("Spearman(token lift, doc lift)")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_png(fig, figs / "summary_by_layer.png", 100)
    plt.close(fig)
    # per-source Q and cross-source overlap (first pool)
    tag = tags[0]
    ps = results[tag]["per_source"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for s, v in ps.items():
        axes[0].plot([v["layers"][str(li)]["Q_spectral"] for li in range(L)], marker="o", label=s)
    axes[0].set_title(f"{tag}: modularity Q per source")
    axes[0].set_xlabel("layer")
    axes[0].legend(fontsize=8)
    xs = results[tag]["cross_source"]
    S = len(xs["sources"])
    J = np.mean(
        [np.array(xs["layers"][str(li)]["top1pct_pair_jaccard"]) for li in range(L)], axis=0
    )
    im = axes[1].imshow(J, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_xticks(range(S))
    axes[1].set_yticks(range(S))
    axes[1].set_xticklabels(xs["sources"], rotation=45, ha="right")
    axes[1].set_yticklabels(xs["sources"])
    for a in range(S):
        for b in range(S):
            axes[1].text(b, a, f"{J[a, b]:.2f}", ha="center", va="center", color="w", fontsize=8)
    axes[1].set_title("Jaccard of top-1% lift pairs between sources (mean over layers)")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    save_png(fig, figs / "per_source.png", 100)
    plt.close(fig)
    # unique experts per doc
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag in tags:
        u = results[tag]["unique_experts_per_doc"]
        ax.plot(u["mean_by_layer"], marker="o", label=f"{tag} mean")
        ax.plot(u["p90_by_layer"], ls="--", label=f"{tag} p90")
    ax.set_xlabel("layer")
    ax.set_ylabel("unique routed experts per document")
    ax.set_title("experts touched per document (docs truncated to 4095 tokens)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_png(fig, figs / "unique_experts_per_doc.png", 100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pools", default="config,64")
    p.add_argument("--k-clusters", type=int, default=8)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    out = args.out or (REPO / "claude_outputs/sparse_experts/coactivation" / args.run_dir.name)
    figs = out / "figs"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag in args.pools.split(","):
        pool_dir = args.run_dir / f"pool_{tag}"
        if not (pool_dir / "MERGED").exists():
            log(f"skip {pool_dir}: not merged")
            continue
        info = json.load(open(pool_dir / "info.json"))
        t0 = time.time()
        results[f"pool_{tag}"] = analyze_pool(
            pool_dir, info["sources"], args.k_clusters, figs, f"pool_{tag}", not args.no_plots
        )
        log(f"analyzed pool_{tag} in {time.time() - t0:.0f}s")
    if not args.no_plots and results:
        make_summary_figures(results, figs)
    json.dump(
        dict(run_dir=str(args.run_dir), k_clusters=args.k_clusters, pools=results),
        open(out / "analysis.json", "w"),
        indent=1,
    )
    log(f"wrote {out / 'analysis.json'}")
    # console digest
    for tag, r in results.items():
        L = len(r["layers"])
        print(
            f"\n== {tag}: {r['num_tokens']:,} tokens, {r['num_docs']:,} docs, eval pool {r['eval_document_expert_pool']}"
        )
        print(
            f"{'layer':>5} {'effExp':>7} {'unused':>6} {'Qspec':>6} {'Qnull':>6} {'Qlouv':>6} {'nComm':>5} {'within':>6} "
            f"{'never%':>6} {'liftMed':>7} {'liftP99':>7} {'rho_td':>6} {'uniq/doc':>8}"
        )
        for li in range(L):
            s = r["layers"][str(li)]["sample"]
            st = s["structure"]
            ql = st["Q_louvain"]
            print(
                f"{li:5d} {s['usage']['effective_experts']:7.1f} {s['usage']['unused']:6d} {st['Q_spectral']:6.3f} "
                f"{st['Q_null_shuffled']:6.3f} {(ql if ql is not None else float('nan')):6.3f} "
                f"{(st['n_louvain_communities'] or 0):5d} {st['within_cluster_mass']:6.2f} "
                f"{100 * s['pairs']['never_coactive_frac']:6.1f} {s['pairs']['lift_median']:7.2f} "
                f"{s['pairs']['lift_p99']:7.2f} {s['token_vs_doc_spearman']:6.2f} "
                f"{r['unique_experts_per_doc']['mean_by_layer'][li]:8.1f}"
            )


if __name__ == "__main__":
    main()
