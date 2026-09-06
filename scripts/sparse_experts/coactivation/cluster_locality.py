#!/usr/bin/env python3
"""
Are the expert clusters seen in the lift heatmaps a *token-level* phenomenon (the same cluster fires
on particular token types in every document) or a *document-level* one (a document spends most of
its routing inside one cluster)?  The answer decides whether a cluster of experts can be finetuned
on a group of documents in isolation, without the rest of the experts in GPU memory.

Uses only the cached per-token routing indices written by extract_coactivation.py (no new forward
passes).  For every layer:

  1. expert clusters = spectral k-way clustering of the token-level lift matrix (same code and
     seed as analyze_coactivation.py, so the clusters are the ones shown in the report heatmaps)
  2. each routed (token, slot) gets the cluster label of its expert; that label is the random
     variable c whose predictability we decompose:
       - given the document           (doc x cluster table)
       - given the token type         (vocab x cluster table)
       - given the k32 topic / source of the document
       - conditional variants I(c; topic | token) and I(c; token | topic)
     all reported as mutual information in bits and as a fraction of H(c), bias-corrected with a
     label-permutation baseline
  3. concentration: share of a document's routing that lands in its single best cluster, compared
     with (a) the global cluster marginal and (b) a cross-fitted "token composition" prediction
     (P(c | token) fitted on the other half of the documents) -- the gap between (b) and the
     actual value is the document effect beyond what its tokens alone explain
  4. sequential structure: top-1 cluster switch rate between consecutive tokens of a document vs
     the within-document shuffle null
  5. feasibility numbers for cluster finetuning: experts needed to cover 90/95/99 % of a
     document's routing, the same per source / per k32 topic group, Jaccard overlap between the
     groups' 95 % expert sets, and the coverage obtained if only a fixed number of clusters is
     loaded (chosen per document, or once per topic group)
  6. top token types per cluster (decoded) for a qualitative look

  python scripts/sparse_experts/coactivation/cluster_locality.py analyze \
      --run-dir sparse_experts/coactivation/sparse_8of512_10b_step2384/pool_config \
      --out claude_outputs/sparse_experts/coactivation/sparse_8of512_10b_step2384/locality \
      --big-out sparse_experts/coactivation/sparse_8of512_10b_step2384/locality

  python scripts/sparse_experts/coactivation/cluster_locality.py plot \
      --run 512=claude_outputs/.../sparse_8of512_10b_step2384/locality/locality.json \
      --run 1024=claude_outputs/.../sparse_8of1024_10b_step2384/locality/locality.json \
      --out claude_outputs/sparse_experts/coactivation/compare/locality
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_coactivation import lift_matrix, modularity, save_png, spectral_labels  # noqa: E402

DOCS_DEFAULT = Path("sparse_experts/coactivation/docs_40k")
THRESHOLDS = (0.9, 0.95, 0.99)
MAX_CLUSTERS_LOADED = 4
COLORS = {"512": "#1f77b4", "1024": "#d62728"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------- inputs


def load_docs(docs_dir: Path, sources: list):
    meta = {}
    for line in open(docs_dir / "doc_meta.jsonl"):
        d = json.loads(line)
        meta[d["doc_id"]] = d
    n = len(meta)
    src = np.zeros(n, np.int64)
    grp = np.zeros(n, np.int64)
    for i, m in meta.items():
        src[i] = sources.index(m["source"])
        grp[i] = m["cluster_k32_vanilla"]
    toks = [None] * n
    with gzip.open(docs_dir / "docs.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            toks[d["doc_id"]] = np.asarray(d["token_ids"], dtype=np.int32)
    return src, grp, toks


def token_ids_in_run_order(tok_doc: np.ndarray, toks: list):
    """topk rows are in packed order and every document is one contiguous run (verified below)."""
    change = np.flatnonzero(np.diff(tok_doc) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(tok_doc)]])
    docs = tok_doc[starts]
    assert len(np.unique(docs)) == len(docs), "a document is split into several runs"
    tokid = np.empty(len(tok_doc), dtype=np.int32)
    for s, e, d in zip(starts, ends, docs):
        t = toks[int(d)]
        assert len(t) >= e - s, (d, len(t), e - s)
        tokid[s:e] = t[: e - s]
    return tokid


def cluster_labels(counts_path: Path, k: int):
    z = np.load(counts_path)
    c_tok = z["c_tok"].sum(0).astype(np.float64)  # (L, E, E) pooled over sources
    N = float(z["n_tok"].sum())
    L, E, _ = c_tok.shape
    Es = E - 1
    labels = np.zeros((L, Es), dtype=np.int64)
    Q = []
    for li in range(L):
        C = c_tok[li, :Es, :Es]
        lift, _ = lift_matrix(C, N)
        lab = spectral_labels(lift, k)
        W = C.copy()
        np.fill_diagonal(W, 0.0)
        labels[li] = lab
        Q.append(modularity(W, lab))
    return labels, Q, Es


# ------------------------------------------------------------------------------------ statistics


def mi_bits(J: np.ndarray):
    """Mutual information (bits) between row and column variables of a joint count table."""
    J = J.astype(np.float64)
    tot = J.sum()
    if tot <= 0:
        return 0.0
    p = J / tot
    pr = p.sum(1, keepdims=True)
    pc = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pr @ pc)[nz])).sum())


def entropy_bits(p: np.ndarray):
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def top_share_and_eff(D: torch.Tensor):
    """Per row: share of the largest column, and exp(entropy) (effective #clusters)."""
    tot = D.sum(1, keepdim=True).clamp_min(1e-12)
    p = D / tot
    top = p.max(1).values
    eff = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum(1))
    return top, eff


def experts_needed(U: torch.Tensor, thresholds):
    """U: (rows, Es) counts. For each threshold, #experts (sorted by count) covering that share."""
    srt = U.sort(1, descending=True).values
    cum = srt.cumsum(1) / srt.sum(1, keepdim=True).clamp_min(1e-12)
    return {str(t): ((cum < t).sum(1) + 1).float() for t in thresholds}


def top_set(u: torch.Tensor, thr: float):
    order = u.argsort(descending=True)
    cum = u[order].cumsum(0) / u.sum().clamp_min(1e-12)
    n = int((cum < thr).sum()) + 1
    return set(order[:n].tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def multinomial_null_distinct(p_e: torch.Tensor, lab: torch.Tensor, k: int, n: int, gen):
    """Distinct-cluster histogram for tokens whose 7 experts are drawn from the usage marginal."""
    hist = torch.zeros(k + 1, dtype=torch.long, device=p_e.device)
    chunk = 200_000
    for _ in range(n // chunk):
        e = torch.multinomial(p_e.expand(chunk, -1), 7, replacement=False, generator=gen)
        cs = lab[e].sort(1).values
        d = 1 + (cs[:, 1:] != cs[:, :-1]).sum(1)
        hist += torch.bincount(d, minlength=k + 1)
    return hist


# --------------------------------------------------------------------------------------- analyze


def analyze(args):
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    out = Path(args.out)
    big = Path(args.big_out) if args.big_out else out
    out.mkdir(parents=True, exist_ok=True)
    big.mkdir(parents=True, exist_ok=True)
    info = json.load(open(run_dir / "info.json"))
    sources = info["sources"]
    k = args.k

    log("clustering lift matrices")
    labels, Q, Es = cluster_labels(run_dir / "counts.npz", k)
    L = labels.shape[0]
    np.save(big / "labels.npy", labels)
    log(f"L={L} Es={Es} k={k} Q={np.round(Q, 3).tolist()}")

    log("loading documents")
    src_doc, grp_doc, toks = load_docs(Path(args.docs), sources)
    ndoc = len(src_doc)
    G = int(grp_doc.max()) + 1
    S = len(sources)
    tok_doc = np.load(run_dir / "tok_doc.npy")
    N = len(tok_doc)
    tokid = token_ids_in_run_order(tok_doc, toks)
    V = int(tokid.max()) + 1
    log(f"N={N:,} tokens, {ndoc} docs, vocab max {V}, {G} topic groups, {S} sources")

    log("reading topk.npy into memory")
    t0 = time.time()
    topk = np.load(run_dir / "topk.npy")  # (N, L, 7) int16
    log(f"read {topk.nbytes / 1e9:.1f} GB in {time.time() - t0:.0f}s")
    assert topk.shape == (N, L, 7), topk.shape

    tok_doc_t = torch.from_numpy(tok_doc.astype(np.int64)).to(device)
    tokid_t = torch.from_numpy(tokid.astype(np.int64)).to(device)
    src_doc_t = torch.from_numpy(src_doc).to(device)
    grp_doc_t = torch.from_numpy(grp_doc).to(device)
    grp_tok_t = grp_doc_t[tok_doc_t]
    doc_len = torch.bincount(tok_doc_t, minlength=ndoc).double()
    present = (
        doc_len > 0
    )  # docs absent from this run (pilot subsets) are excluded from per-doc stats
    even_doc = (tok_doc_t % 2) == 0
    gen = torch.Generator(device=device).manual_seed(0)
    perm = torch.randperm(N * 7, device=device, generator=gen)

    doc_cluster = np.zeros((L, ndoc, k), dtype=np.int64)
    doc_expert = np.lib.format.open_memmap(
        big / "doc_expert.npy", mode="w+", dtype=np.int32, shape=(L, ndoc, Es)
    )
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    except Exception as ex:  # noqa: BLE001
        log(f"tokenizer unavailable ({ex}); top tokens will be ids only")

    layers = {}
    for li in range(L):
        t0 = time.time()
        e = torch.from_numpy(topk[:, li, :].astype(np.int64)).to(device)  # (N, 7)
        lab = torch.from_numpy(labels[li]).to(device)
        c = lab[e]  # (N, 7)
        cf = c.reshape(-1)
        d7 = tok_doc_t.repeat_interleave(7)
        v7 = tokid_t.repeat_interleave(7)
        g7 = grp_tok_t.repeat_interleave(7)
        r = {}

        # ---- 1. cluster bookkeeping
        sizes = torch.bincount(lab, minlength=k)
        mass = torch.bincount(cf, minlength=k).double()
        r["clusters"] = dict(
            sizes=sizes.tolist(),
            mass_share=(mass / mass.sum()).tolist(),
            Q_spectral=Q[li],
            H_cluster_bits=entropy_bits(mass.cpu().numpy()),
        )

        # ---- 2. per-token spread over clusters
        cs = c.sort(1).values
        distinct = 1 + (cs[:, 1:] != cs[:, :-1]).sum(1)
        hist = torch.bincount(distinct, minlength=k + 1)[1:8].double()
        p_e = torch.bincount(e.reshape(-1), minlength=Es).double()
        p_e = (p_e / p_e.sum()).float()
        hist_null = multinomial_null_distinct(p_e, lab, k, 1_000_000, gen)[1:8].double()
        r["per_token"] = dict(
            distinct_clusters_mean=float(
                (hist * torch.arange(1, 8, device=device)).sum() / hist.sum()
            ),
            distinct_clusters_hist=(hist / hist.sum()).tolist(),
            null_distinct_clusters_mean=float(
                (hist_null * torch.arange(1, 8, device=device)).sum() / hist_null.sum()
            ),
            null_distinct_clusters_hist=(hist_null / hist_null.sum()).tolist(),
        )

        # ---- 3. joint tables
        D = torch.bincount(d7 * k + cf, minlength=ndoc * k).view(ndoc, k).double()
        T = torch.bincount(v7 * k + cf, minlength=V * k).view(V, k).double()
        TG = torch.bincount((v7 * G + g7) * k + cf, minlength=V * G * k).view(V * G, k).double()
        SG = torch.zeros(S, k, dtype=torch.float64, device=device).index_add_(0, src_doc_t, D)
        GG = torch.zeros(G, k, dtype=torch.float64, device=device).index_add_(0, grp_doc_t, D)
        U = torch.bincount(d7 * Es + e.reshape(-1), minlength=ndoc * Es).view(ndoc, Es)
        doc_cluster[li] = D.cpu().numpy()
        doc_expert[li] = U.cpu().numpy().astype(np.int32)

        # permutation baseline for MI bias (labels shuffled across all slots)
        cp = cf[perm]
        Dp = torch.bincount(d7 * k + cp, minlength=ndoc * k).view(ndoc, k)
        Tp = torch.bincount(v7 * k + cp, minlength=V * k).view(V, k)
        TGp = torch.bincount((v7 * G + g7) * k + cp, minlength=V * G * k).view(V * G, k)
        GGp = torch.zeros(G, k, dtype=torch.float64, device=device).index_add_(
            0, grp_doc_t, Dp.double()
        )
        SGp = torch.zeros(S, k, dtype=torch.float64, device=device).index_add_(
            0, src_doc_t, Dp.double()
        )

        def mi(J, Jp):
            raw = mi_bits(J.cpu().numpy())
            bias = mi_bits(Jp.cpu().numpy())
            return dict(raw=raw, bias=bias, corrected=max(raw - bias, 0.0))

        Hc = r["clusters"]["H_cluster_bits"]
        I_doc = mi(D, Dp)
        I_tok = mi(T, Tp)
        I_tokgrp = mi(TG, TGp)
        I_grp = mi(GG, GGp)
        I_src = mi(SG, SGp)
        I_grp_given_tok = max(I_tokgrp["corrected"] - I_tok["corrected"], 0.0)
        I_tok_given_grp = max(I_tokgrp["corrected"] - I_grp["corrected"], 0.0)
        r["mutual_information_bits"] = dict(
            H_cluster=Hc,
            I_doc=I_doc,
            I_token=I_tok,
            I_topic=I_grp,
            I_source=I_src,
            I_token_and_topic=I_tokgrp,
            I_topic_given_token=I_grp_given_tok,
            I_token_given_topic=I_tok_given_grp,
        )
        r["mutual_information_frac"] = {
            "doc": I_doc["corrected"] / Hc,
            "token": I_tok["corrected"] / Hc,
            "topic": I_grp["corrected"] / Hc,
            "source": I_src["corrected"] / Hc,
            "token_and_topic": I_tokgrp["corrected"] / Hc,
            "topic_given_token": I_grp_given_tok / Hc,
            "token_given_topic": I_tok_given_grp / Hc,
        }

        # ---- 4. concentration: actual per doc vs marginal vs token-composition (cross-fitted)
        top_d, eff_d = top_share_and_eff(D[present])
        marg = mass / mass.sum()
        pred = torch.zeros(ndoc, k, dtype=torch.float64, device=device)
        for fit_mask, apply_mask in [(even_doc, ~even_doc), (~even_doc, even_doc)]:
            Tf = (
                torch.bincount(
                    v7[fit_mask.repeat_interleave(7)] * k + cf[fit_mask.repeat_interleave(7)],
                    minlength=V * k,
                )
                .view(V, k)
                .double()
            )
            rs = Tf.sum(1, keepdim=True)
            Pf = torch.where(rs > 0, Tf / rs.clamp_min(1), marg[None, :])
            rows = Pf[tokid_t[apply_mask]]
            pred.index_add_(0, tok_doc_t[apply_mask], rows)
        top_p, eff_p = top_share_and_eff(pred[present])
        src_present = src_doc_t[present]
        # token-type concentration (given the token, how peaked is the cluster distribution)
        tot_T = T.sum(1)
        keep = tot_T >= 7 * args.min_token_count
        top_t, eff_t = top_share_and_eff(T[keep])
        w = tot_T[keep] / tot_T[keep].sum()
        r["concentration"] = dict(
            doc_top_share_mean=float(top_d.mean()),
            doc_top_share_median=float(top_d.median()),
            doc_top_share_p10=float(torch.quantile(top_d, 0.1)),
            doc_top_share_p90=float(torch.quantile(top_d, 0.9)),
            doc_eff_clusters_mean=float(eff_d.mean()),
            token_null_top_share_mean=float(top_p.mean()),
            token_null_eff_clusters_mean=float(eff_p.mean()),
            marginal_top_share=float(marg.max()),
            marginal_eff_clusters=float(np.exp(entropy_bits(marg.cpu().numpy()) * np.log(2))),
            token_type_top_share_mean=float((top_t * w).sum()),
            token_type_eff_clusters_mean=float((eff_t * w).sum()),
            token_type_slot_coverage=float(tot_T[keep].sum() / tot_T.sum()),
            doc_top_share_by_source={
                s: float(top_d[src_present == si].mean()) for si, s in enumerate(sources)
            },
            token_null_top_share_by_source={
                s: float(top_p[src_present == si].mean()) for si, s in enumerate(sources)
            },
        )
        r["group_cluster_share"] = dict(
            topic=(GG / GG.sum(1, keepdim=True)).cpu().numpy().round(4).tolist(),
            source=(SG / SG.sum(1, keepdim=True)).cpu().numpy().round(4).tolist(),
        )

        # ---- 5. sequential structure (top-1 slot)
        c0 = c[:, 0]
        same = tok_doc_t[1:] == tok_doc_t[:-1]
        sw = ((c0[1:] != c0[:-1]) & same).sum().double() / same.sum().double()
        D0 = torch.bincount(tok_doc_t * k + c0, minlength=ndoc * k).view(ndoc, k).double()
        p0 = D0 / D0.sum(1, keepdim=True).clamp_min(1)
        null_sw = (((1 - (p0**2).sum(1)) * (doc_len - 1).clamp_min(0)).sum()) / (
            (doc_len - 1).clamp_min(0).sum()
        )
        r["sequential"] = dict(
            top1_switch_rate=float(sw),
            top1_switch_rate_within_doc_shuffle=float(null_sw),
            top1_mean_run_length=float(1.0 / max(float(sw), 1e-9)),
            top1_mean_run_length_null=float(1.0 / max(float(null_sw), 1e-9)),
        )

        # ---- 6. expert coverage per doc / per group
        Uf = U.double()
        need = experts_needed(Uf[present], THRESHOLDS)
        Ug = torch.zeros(G, Es, dtype=torch.float64, device=device).index_add_(0, grp_doc_t, Uf)
        Us = torch.zeros(S, Es, dtype=torch.float64, device=device).index_add_(0, src_doc_t, Uf)
        need_g = experts_needed(Ug, THRESHOLDS)
        need_s = experts_needed(Us, THRESHOLDS)
        sets_g = [top_set(Ug[g], 0.95) for g in range(G)]
        sets_s = [top_set(Us[s], 0.95) for s in range(S)]
        jac_g = np.array([[jaccard(a, b) for b in sets_g] for a in sets_g])
        jac_s = np.array([[jaccard(a, b) for b in sets_s] for a in sets_s])
        off = ~np.eye(G, dtype=bool)
        # union of the groups' 95% sets, i.e. how much of the model any single group leaves unused
        r["coverage"] = dict(
            per_doc_experts_needed={
                t: dict(
                    mean=float(v.mean()),
                    median=float(v.median()),
                    frac_of_experts_mean=float(v.mean() / Es),
                )
                for t, v in need.items()
            },
            per_topic_experts_needed={
                t: dict(mean=float(v.mean()), frac_of_experts_mean=float(v.mean() / Es))
                for t, v in need_g.items()
            },
            per_source_experts_needed={
                t: dict(
                    by_source={s: float(v[si]) for si, s in enumerate(sources)},
                    frac_of_experts_mean=float(v.mean() / Es),
                )
                for t, v in need_s.items()
            },
            topic_jaccard_95_mean=float(jac_g[off].mean()),
            topic_jaccard_95_min=float(jac_g[off].min()),
            topic_jaccard_95_max=float(jac_g[off].max()),
            topic_jaccard_95=jac_g.round(3).tolist(),
            source_jaccard_95=jac_s.round(3).tolist(),
            unused_by_topic_frac_mean=float(np.mean([1 - len(s) / Es for s in sets_g])),
        )

        # cluster-restricted coverage: load only m whole clusters
        Dn = D / D.sum(1, keepdim=True).clamp_min(1)
        Dsorted = Dn.sort(1, descending=True).values.cumsum(1)
        doc_choice = {
            str(m): float(Dsorted[present, m - 1].mean()) for m in range(1, MAX_CLUSTERS_LOADED + 1)
        }
        size_frac = sizes.double() / Es
        grp_choice, grp_mem = {}, {}
        Gn = GG / GG.sum(1, keepdim=True)
        for m in range(1, MAX_CLUSTERS_LOADED + 1):
            top_c = Gn.argsort(1, descending=True)[:, :m]  # (G, m)
            cov = torch.zeros(ndoc, dtype=torch.float64, device=device)
            mem = torch.zeros(G, dtype=torch.float64, device=device)
            for g in range(G):
                sel = top_c[g]
                cov[grp_doc_t == g] = Dn[grp_doc_t == g][:, sel].sum(1)
                mem[g] = size_frac[sel].sum()
            grp_choice[str(m)] = float(cov[present].mean())
            grp_mem[str(m)] = float(mem.mean())
        r["cluster_restricted"] = dict(
            doc_specific_coverage=doc_choice,
            topic_fixed_coverage=grp_choice,
            topic_fixed_memory_frac=grp_mem,
            equal_cluster_memory_frac={str(m): m / k for m in range(1, MAX_CLUSTERS_LOADED + 1)},
        )

        # ---- 7. top token types per cluster
        Pn = torch.where(tot_T[:, None] > 0, T / tot_T[:, None].clamp_min(1), marg[None, :])
        lift_tc = Pn / marg[None, :]
        lift_tc[~keep] = 0
        top_tokens = {}
        for ci in range(k):
            vals, idx = lift_tc[:, ci].topk(args.n_top_tokens)
            ids = idx.tolist()
            strs = [tokenizer.decode([i]) for i in ids] if tokenizer else [str(i) for i in ids]
            top_tokens[str(ci)] = [
                dict(token=s, id=i, lift=round(float(v), 2), p=round(float(Pn[i, ci]), 3))
                for s, i, v in zip(strs, ids, vals.tolist())
            ]
        r["top_tokens_per_cluster"] = top_tokens

        layers[str(li)] = r
        m = r["mutual_information_frac"]
        cc = r["concentration"]
        log(
            f"layer {li:2d} ({time.time() - t0:.0f}s) Q={Q[li]:.3f} H={Hc:.2f}b "
            f"I/H doc={m['doc']:.3f} tok={m['token']:.3f} topic={m['topic']:.3f} "
            f"topic|tok={m['topic_given_token']:.3f} | top-share doc={cc['doc_top_share_mean']:.3f} "
            f"tokNull={cc['token_null_top_share_mean']:.3f} marg={cc['marginal_top_share']:.3f} "
            f"tokType={cc['token_type_top_share_mean']:.3f} | switch={r['sequential']['top1_switch_rate']:.3f}"
            f"/{r['sequential']['top1_switch_rate_within_doc_shuffle']:.3f} | "
            f"exp95 doc={need['0.95'].mean():.0f} topic={need_g['0.95'].mean():.0f} "
            f"J95={jac_g[off].mean():.2f}"
        )
        del e, c, cf, d7, v7, g7, D, T, TG, U, Dp, Tp, TGp, pred, Uf, cp
        torch.cuda.empty_cache()

    doc_expert.flush()
    del doc_expert
    np.save(big / "doc_cluster.npy", doc_cluster)
    res = dict(
        run_dir=str(run_dir),
        docs=str(args.docs),
        num_tokens=N,
        num_docs=ndoc,
        num_std_experts=Es,
        num_layers=L,
        k=k,
        sources=sources,
        num_topics=G,
        min_token_count=args.min_token_count,
        layers=layers,
    )
    json.dump(res, open(out / "locality.json", "w"), indent=1)
    log(f"wrote {out / 'locality.json'}")


# ------------------------------------------------------------------------------------------ plot


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = json.load(open(path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def col(label):
        return COLORS.get(label)

    def ser(res, fn):
        return [fn(res["layers"][str(li)]) for li in range(res["num_layers"])]

    # 1) concentration: top-cluster share, per model panel
    fig, axes = plt.subplots(2, len(runs), figsize=(6.5 * len(runs), 8), squeeze=False)
    for j, (label, res) in enumerate(runs.items()):
        ax = axes[0, j]
        c = lambda key: ser(res, lambda r: r["concentration"][key])  # noqa: E731
        ax.plot(
            c("doc_top_share_mean"),
            marker="o",
            color="k",
            lw=2,
            label="given the document (actual)",
        )
        ax.fill_between(
            range(res["num_layers"]),
            c("doc_top_share_p10"),
            c("doc_top_share_p90"),
            color="k",
            alpha=0.12,
            label="doc p10-p90",
        )
        ax.plot(
            c("token_null_top_share_mean"),
            marker="s",
            color="#ff7f0e",
            label="predicted from the doc's token types only (cross-fitted)",
        )
        ax.plot(
            c("token_type_top_share_mean"),
            marker="^",
            color="#2ca02c",
            label="given the token type (freq-weighted)",
        )
        ax.plot(c("marginal_top_share"), ls=":", color="gray", label="global marginal (no info)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} experts: share of routing in the single best cluster")
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        ax.plot(c("doc_eff_clusters_mean"), marker="o", color="k", lw=2, label="given the document")
        ax.plot(
            c("token_null_eff_clusters_mean"),
            marker="s",
            color="#ff7f0e",
            label="token-composition prediction",
        )
        ax.plot(
            c("token_type_eff_clusters_mean"),
            marker="^",
            color="#2ca02c",
            label="given the token type",
        )
        ax.plot(c("marginal_eff_clusters"), ls=":", color="gray", label="global marginal")
        ax.set_ylim(1, res["k"] + 0.2)
        ax.set_title(f"{label} experts: effective number of clusters (exp H)")
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, out / "locality_concentration.png", 100)
    plt.close(fig)

    # 2) mutual information fractions
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4.5), squeeze=False)
    keys = [
        ("token", "token type", "#2ca02c", "-"),
        ("doc", "document", "k", "-"),
        ("topic", "k32 topic of the doc", "#9467bd", "-"),
        ("source", "source of the doc", "#8c564b", "-"),
        ("topic_given_token", "topic | token type", "#9467bd", "--"),
        ("token_given_topic", "token type | topic", "#2ca02c", "--"),
    ]
    for j, (label, res) in enumerate(runs.items()):
        ax = axes[0, j]
        for key, name, color, ls in keys:
            ax.plot(
                ser(res, lambda r: r["mutual_information_frac"][key]),
                marker="o",
                ms=4,
                color=color,
                ls=ls,
                label=name,
            )
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} experts: I(cluster; X) / H(cluster)")
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, out / "locality_mi.png", 100)
    plt.close(fig)

    # 3) topic x cluster share heatmaps
    show = [0, 3, 7, 11, 15]
    fig, axes = plt.subplots(
        len(runs), len(show), figsize=(3.2 * len(show), 5.5 * len(runs)), squeeze=False
    )
    for i, (label, res) in enumerate(runs.items()):
        for j, li in enumerate(show):
            M = np.array(res["layers"][str(li)]["group_cluster_share"]["topic"])
            # order clusters by size of their largest topic share for readability
            im = axes[i, j].imshow(M, cmap="magma", vmin=0, vmax=0.8, aspect="auto")
            axes[i, j].set_title(f"{label} experts, layer {li}", fontsize=9)
            axes[i, j].set_xlabel("expert cluster")
            if j == 0:
                axes[i, j].set_ylabel("k32 topic group of the document")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="share of the topic's routing")
    save_png(fig, out / "locality_topic_cluster.png", 100)
    plt.close(fig)

    # 4) coverage / feasibility
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    for label, res in runs.items():
        for t, ls in [("0.9", ":"), ("0.95", "-"), ("0.99", "--")]:
            axes[0].plot(
                ser(
                    res,
                    lambda r: r["coverage"]["per_doc_experts_needed"][t]["frac_of_experts_mean"],
                ),
                ls=ls,
                marker="o",
                ms=3,
                color=col(label),
                label=f"{label}, {float(t):.0%}",
            )
            axes[1].plot(
                ser(
                    res,
                    lambda r: r["coverage"]["per_topic_experts_needed"][t]["frac_of_experts_mean"],
                ),
                ls=ls,
                marker="o",
                ms=3,
                color=col(label),
                label=f"{label}, {float(t):.0%}",
            )
        axes[2].plot(
            ser(res, lambda r: r["coverage"]["topic_jaccard_95_mean"]),
            marker="o",
            color=col(label),
            label=f"{label} mean over topic pairs",
        )
        axes[2].fill_between(
            range(res["num_layers"]),
            ser(res, lambda r: r["coverage"]["topic_jaccard_95_min"]),
            ser(res, lambda r: r["coverage"]["topic_jaccard_95_max"]),
            color=col(label),
            alpha=0.15,
        )
        for m, ls in [("1", "-"), ("2", "--")]:
            axes[3].plot(
                ser(res, lambda r: r["cluster_restricted"]["topic_fixed_coverage"][m]),
                ls=ls,
                marker="o",
                ms=3,
                color=col(label),
                label=f"{label}: {m} cluster(s) fixed per topic",
            )
            axes[3].plot(
                ser(res, lambda r: r["cluster_restricted"]["doc_specific_coverage"][m]),
                ls=ls,
                marker="x",
                ms=4,
                color=col(label),
                alpha=0.45,
                label=f"{label}: {m} best cluster(s) per doc",
            )
    axes[0].set_title("experts needed per document (fraction of all)")
    axes[1].set_title("experts needed per k32 topic group (fraction of all)")
    axes[2].set_title("Jaccard of 95%-expert sets between topic groups")
    axes[3].set_title("routing covered when only whole clusters are loaded")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[3].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, out / "locality_coverage.png", 100)
    plt.close(fig)

    # 5) sequential + per-token spread
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, res in runs.items():
        axes[0].plot(
            ser(res, lambda r: r["sequential"]["top1_switch_rate"]),
            marker="o",
            color=col(label),
            label=f"{label} actual",
        )
        axes[0].plot(
            ser(res, lambda r: r["sequential"]["top1_switch_rate_within_doc_shuffle"]),
            marker="o",
            ls="--",
            color=col(label),
            alpha=0.6,
            label=f"{label} within-doc shuffle",
        )
        axes[1].plot(
            ser(res, lambda r: r["per_token"]["distinct_clusters_mean"]),
            marker="o",
            color=col(label),
            label=f"{label} actual",
        )
        axes[1].plot(
            ser(res, lambda r: r["per_token"]["null_distinct_clusters_mean"]),
            marker="o",
            ls="--",
            color=col(label),
            alpha=0.6,
            label=f"{label} usage-marginal null",
        )
    axes[0].set_title("top-1 cluster switch rate between consecutive tokens")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("distinct clusters among a token's 7 routed experts")
    axes[1].set_ylim(1, 7)
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_png(fig, out / "locality_sequential.png", 100)
    plt.close(fig)

    # 6) topic jaccard heatmaps at a mid layer + source jaccard
    li = args.jaccard_layer
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 5), squeeze=False)
    for j, (label, res) in enumerate(runs.items()):
        J = np.array(res["layers"][str(li)]["coverage"]["topic_jaccard_95"])
        im = axes[0, j].imshow(J, cmap="viridis", vmin=0, vmax=1)
        axes[0, j].set_title(f"{label} experts, layer {li}: Jaccard of topic 95%-expert sets")
        axes[0, j].set_xlabel("k32 topic")
        axes[0, j].set_ylabel("k32 topic")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    save_png(fig, out / "locality_topic_jaccard.png", 100)
    plt.close(fig)

    # digest
    digest = {}
    for label, res in runs.items():
        L = res["num_layers"]
        lay = [res["layers"][str(li)] for li in range(L)]

        def mean(fn, lo=0):
            return float(np.mean([fn(r) for r in lay[lo:]]))

        digest[label] = dict(
            num_std_experts=res["num_std_experts"],
            doc_top_share=mean(lambda r: r["concentration"]["doc_top_share_mean"]),
            token_null_top_share=mean(lambda r: r["concentration"]["token_null_top_share_mean"]),
            token_type_top_share=mean(lambda r: r["concentration"]["token_type_top_share_mean"]),
            marginal_top_share=mean(lambda r: r["concentration"]["marginal_top_share"]),
            mi_frac={
                key: mean(lambda r, key=key: r["mutual_information_frac"][key])
                for key in [
                    "token",
                    "doc",
                    "topic",
                    "source",
                    "topic_given_token",
                    "token_given_topic",
                ]
            },
            switch_rate=mean(lambda r: r["sequential"]["top1_switch_rate"]),
            switch_rate_null=mean(lambda r: r["sequential"]["top1_switch_rate_within_doc_shuffle"]),
            distinct_clusters_per_token=mean(lambda r: r["per_token"]["distinct_clusters_mean"]),
            distinct_clusters_null=mean(lambda r: r["per_token"]["null_distinct_clusters_mean"]),
            experts95_per_doc_frac=mean(
                lambda r: r["coverage"]["per_doc_experts_needed"]["0.95"]["frac_of_experts_mean"]
            ),
            experts95_per_topic_frac=mean(
                lambda r: r["coverage"]["per_topic_experts_needed"]["0.95"]["frac_of_experts_mean"]
            ),
            experts99_per_topic_frac=mean(
                lambda r: r["coverage"]["per_topic_experts_needed"]["0.99"]["frac_of_experts_mean"]
            ),
            topic_jaccard95=mean(lambda r: r["coverage"]["topic_jaccard_95_mean"]),
            topic_fixed_1cluster_coverage=mean(
                lambda r: r["cluster_restricted"]["topic_fixed_coverage"]["1"]
            ),
            topic_fixed_2cluster_coverage=mean(
                lambda r: r["cluster_restricted"]["topic_fixed_coverage"]["2"]
            ),
            doc_best_1cluster_coverage=mean(
                lambda r: r["cluster_restricted"]["doc_specific_coverage"]["1"]
            ),
        )
    json.dump(digest, open(out / "digest.json", "w"), indent=1)
    print(json.dumps(digest, indent=1))


# ---------------------------------------------------------------------------------------- growth


def growth(args):
    """How the 95%-coverage expert set grows with the number of documents pooled together, for
    documents drawn from one k32 topic, from one source, or at random (no grouping)."""
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    big = Path(args.big_out)
    out = Path(args.out)
    info = json.load(open(run_dir / "info.json"))
    sources = info["sources"]
    src_doc, grp_doc, _ = load_docs(Path(args.docs), sources)
    U_all = np.load(big / "doc_expert.npy", mmap_mode="r")  # (L, ndoc, Es)
    L, ndoc, Es = U_all.shape
    rng = np.random.default_rng(0)
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    groupings = {
        "random": [np.arange(ndoc)],
        "same source": [np.flatnonzero(src_doc == s) for s in range(len(sources))],
        "same k32 topic": [np.flatnonzero(grp_doc == g) for g in range(int(grp_doc.max()) + 1)],
    }
    res = {}
    for li in range(L):
        U = torch.from_numpy(np.ascontiguousarray(U_all[li])).to(device).double()
        r = {}
        for name, pools in groupings.items():
            curve = []
            for n in sizes:
                vals = []
                for _ in range(args.draws):
                    pool = pools[rng.integers(len(pools))]
                    if len(pool) < n:
                        continue
                    idx = torch.from_numpy(rng.choice(pool, n, replace=False)).to(device)
                    u = U[idx].sum(0)
                    vals.append(len(top_set(u, args.threshold)) / Es)
                curve.append(float(np.mean(vals)) if vals else None)
            r[name] = curve
        # per-source curves (each source separately) for the table
        for si, s in enumerate(sources):
            pool = np.flatnonzero(src_doc == si)
            curve = []
            for n in sizes:
                vals = []
                for _ in range(args.draws // 2):
                    idx = torch.from_numpy(rng.choice(pool, n, replace=False)).to(device)
                    vals.append(len(top_set(U[idx].sum(0), args.threshold)) / Es)
                curve.append(float(np.mean(vals)))
            r[f"source:{s}"] = curve
        res[str(li)] = r
        log(f"layer {li}: " + " | ".join(f"{k} {[round(v, 2) for v in r[k]]}" for k in groupings))
    json.dump(
        dict(sizes=sizes, threshold=args.threshold, layers=res, sources=sources),
        open(out / "growth.json", "w"),
        indent=1,
    )
    log(f"wrote {out / 'growth.json'}")


def plot_growth(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = json.load(open(path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    show = [0, 7, 15]
    fig, axes = plt.subplots(
        len(runs), len(show), figsize=(5.2 * len(show), 4.2 * len(runs)), squeeze=False
    )
    for i, (label, g) in enumerate(runs.items()):
        for j, li in enumerate(show):
            ax = axes[i, j]
            r = g["layers"][str(li)]
            for name, color in [
                ("random", "gray"),
                ("same k32 topic", "#9467bd"),
                ("same source", "#8c564b"),
            ]:
                ax.plot(g["sizes"], r[name], marker="o", color=color, label=name)
            for s, color in zip(
                g["sources"], ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf"]
            ):
                ax.plot(g["sizes"], r[f"source:{s}"], ls=":", color=color, label=f"only {s}")
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1)
            ax.set_xlabel("documents pooled")
            ax.set_ylabel(f"experts for {g['threshold']:.0%} coverage (fraction)")
            ax.set_title(f"{label} experts, layer {li}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    save_png(fig, out / "locality_growth.png", 100)
    plt.close(fig)
    print("wrote", out / "locality_growth.png")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument(
        "--run-dir", required=True, help="pool_config dir with topk.npy/tok_doc.npy/counts.npz"
    )
    a.add_argument("--docs", default=str(DOCS_DEFAULT))
    a.add_argument("--out", required=True)
    a.add_argument(
        "--big-out", default=None, help="where to put labels/doc_expert/doc_cluster .npy"
    )
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--min-token-count", type=int, default=100)
    a.add_argument("--n-top-tokens", type=int, default=20)
    a.add_argument("--device", default="cuda")
    a.set_defaults(fn=analyze)
    b = sub.add_parser("plot")
    b.add_argument("--run", action="append", required=True, help="label=path/to/locality.json")
    b.add_argument("--out", required=True)
    b.add_argument("--jaccard-layer", type=int, default=7)
    b.set_defaults(fn=plot)
    g = sub.add_parser("growth")
    g.add_argument("--run-dir", required=True)
    g.add_argument("--big-out", required=True, help="dir holding doc_expert.npy from analyze")
    g.add_argument("--docs", default=str(DOCS_DEFAULT))
    g.add_argument("--out", required=True)
    g.add_argument("--threshold", type=float, default=0.95)
    g.add_argument("--draws", type=int, default=40)
    g.add_argument("--device", default="cuda")
    g.set_defaults(fn=growth)
    pg = sub.add_parser("plot-growth")
    pg.add_argument("--run", action="append", required=True, help="label=path/to/growth.json")
    pg.add_argument("--out", required=True)
    pg.set_defaults(fn=plot_growth)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
