#!/usr/bin/env python3
"""Router weights and logits after disjoint training (user request 2026-11-06): reads the router_logits.py dumps
(sparse_experts/olmoe3_routing/runs_routerlogits/<model>/<tag>/router_logits.npz) of the start model, the joint baseline, the merged
4- and 8-square models at sub-model LR 4e-4 and every single square of EMO 512e / 1000e / 2000e, and writes
claude_outputs/olmoe3_routing/router_logits.json: per model, K (4 | 8), matched point and MoE layer,

  for each full model (start / joint / merged), with the experts grouped by the K-way split:
    row_norm           mean L2 norm of the router rows (and per group: row_norm_g)
    centroid_g         norm of each group's mean router row (a direction shared by a group = one logit offset for the whole group)
    logit_std          mean over tokens of the std of the logits across ALL experts (the logit scale the softmax sees)
    within_std         mean over tokens and groups of the std across the experts of ONE group
    offset_std         mean over tokens of the std ACROSS groups of the group-mean logit: the per-token offset between the groups
    offset_ratio       offset_std / (within_std / sqrt(experts per group)): 1 = what random groups of one joint router give
    top_share_g        share of the top-16 slots that go to each group (1/K at balance)
    token_max_share    mean over tokens of the largest group's share of the token's top-16 (1/K .. 1)
    top1, top16, z     mean largest logit, mean 16th-largest logit, mean logsumexp
    topk_mass          mean softmax mass of the top-16
    q_g                per group, the 1 / 5 / 25 / 50 / 75 / 95 / 99 % quantiles of its experts' logits (all sampled tokens)
  for the squares on their own (mean over the K squares, and per square):
    sq_row_norm_g, sq_logit_std (std across the square's own experts), sq_z, sq_top1,
    sq_vs_merged_corr  correlation between a square's own logits and the merged model's logits for the same experts and tokens
                       (per-token centred within the group): how much the averaged trunk changes what the router rows see

Layer 1 is whole in every square (not partitioned): its "groups" are the layer-2 groups, as a random reference grouping.
  python scripts/olmoe3_routing/router_logit_analysis.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "sparse_experts"; R = S / "olmoe3_routing/runs_routerlogits"; OUT = ROOT / "claude_outputs/olmoe3_routing"
LAYERS = list(range(1, 10)); POINTS = [19074, 30000, 38148, 57221, 116479]; QS = [1, 5, 25, 50, 75, 95, 99]
ARMS = {"emo512": {4: "olmoe3_squares_emorand_lr4e-4", 8: "olmoe3_squares_emorand8_lr4e-4"},
        "emo1000": {4: "olmoe3_squares_emo1000rand4", 8: "olmoe3_squares_emo1000rand8"},
        "emo2000": {4: "olmoe3_squares_emo2000rand4", 8: "olmoe3_squares_emo2000rand8"}}
TOP = 16


def load(model, tag):
    f = R / model / tag / "router_logits.npz"
    return np.load(f) if f.exists() and (f.parent / "meta.json").exists() else None


def groups_of(sqn, K):
    G = json.load(open(S / sqn / "groups.json"))["groups"]
    whole = lambda gl: len(set(gl[0])) == len(set().union(*gl))   # a layer kept whole lists ALL its experts in every group (layer 1)
    g = {l: [np.asarray(G[str(l)][k], dtype=np.int64) for k in range(K)] for l in LAYERS if str(l) in G and len(G[str(l)]) == K and not whole(G[str(l)])}
    ref = g[2]
    return {l: g.get(l, ref) for l in LAYERS}, [l for l in LAYERS if l in g]   # groups per layer (layer 1: layer-2 groups as reference), partitioned layers


def full_stats(z, grp):
    out = {}
    for l in LAYERS:
        W = z[f"weight_L{l}"].astype(np.float32); X = z[f"logits_L{l}"].astype(np.float32)        # (E, d), (n, E)
        gs = grp[l]; K = len(gs); n = X.shape[0]
        gm = np.stack([X[:, g].mean(1) for g in gs], 1); gsd = np.stack([X[:, g].std(1) for g in gs], 1)          # (n, K)
        within = float(gsd.mean()); off = float(gm.std(1).mean()); per = np.mean([len(g) for g in gs])
        order = np.argsort(-X, axis=1)[:, :TOP]; e2g = np.full(X.shape[1], -1); [e2g.__setitem__(g, k) for k, g in enumerate(gs)]
        tg = e2g[order]; cnt = np.stack([(tg == k).sum(1) for k in range(K)], 1)                                     # (n, K)
        srt = np.take_along_axis(X, order, 1); mx = X.max(1, keepdims=True); lse = (mx[:, 0] + np.log(np.exp(X - mx).sum(1)))
        p = np.exp(X - lse[:, None])
        out[str(l)] = dict(row_norm=float(np.linalg.norm(W, axis=1).mean()), row_norm_g=[float(np.linalg.norm(W[g], axis=1).mean()) for g in gs],
                           centroid_g=[float(np.linalg.norm(W[g].mean(0))) for g in gs], logit_mean=float(X.mean()), logit_std=float(X.std(1).mean()), within_std=within,
                           offset_std=off, offset_ratio=off / (within / np.sqrt(per)), top_share_g=[float(c) for c in cnt.sum(0) / cnt.sum()],
                           token_max_share=float((cnt.max(1) / TOP).mean()), top1=float(srt[:, 0].mean()), top16=float(srt[:, TOP - 1].mean()), z=float(lse.mean()),
                           topk_mass=float(np.take_along_axis(p, order, 1).sum(1).mean()), q_g=[[float(v) for v in np.percentile(X[:, g], QS)] for g in gs],
                           group_mean_g=[float(v) for v in gm.mean(0)])
    return out


def square_stats(zs, zm, grp, part):
    """zs: the K squares' dumps (own expert order = groups.json order); zm: the merged model at the same point (or None)."""
    out = {}
    for l in LAYERS:
        rows, sds, zz, t1, cor, qq = [], [], [], [], [], []
        for k, z in enumerate(zs):
            W = z[f"weight_L{l}"].astype(np.float32); X = z[f"logits_L{l}"].astype(np.float32)
            mx = X.max(1, keepdims=True)
            qq.append([float(v) for v in np.percentile(X, QS)])
            rows.append(float(np.linalg.norm(W, axis=1).mean())); sds.append(float(X.std(1).mean())); zz.append(float((mx[:, 0] + np.log(np.exp(X - mx).sum(1))).mean())); t1.append(float(mx.mean()))
            if zm is not None and l in part:
                M = zm[f"logits_L{l}"].astype(np.float32)[:, grp[l][k]]
                a = X - X.mean(1, keepdims=True); b = M - M.mean(1, keepdims=True)
                cor.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())))
        out[str(l)] = dict(sq_q_g=qq, sq_row_norm_g=rows, sq_logit_std_g=sds, sq_z_g=zz, sq_top1_g=t1, sq_vs_merged_corr_g=cor,
                           sq_row_norm=float(np.mean(rows)), sq_logit_std=float(np.mean(sds)), sq_z=float(np.mean(zz)), sq_vs_merged_corr=float(np.mean(cor)) if cor else None)
    return out


def figures(res):
    """One box-plot figure per model, K and point: per layer, the distribution (5 / 25 / 50 / 75 / 95 % of all sampled tokens x experts) of the logits
    of each square's experts in the joint baseline (grey), in the merged model (colour) and in the square's own model (outline)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG = OUT / "router_logits_figs"; FIG.mkdir(parents=True, exist_ok=True); n = 0
    box = lambda q, pos: dict(med=q[3], q1=q[2], q3=q[4], whislo=q[1], whishi=q[5], fliers=[])
    for model, arms in res.items():
        for K, r in arms.items():
            for pt, e in r["points"].items():
                if "merged" not in e: continue
                K_ = int(K); fig, axes = plt.subplots(1, 9, figsize=(2.3 * 9, 3.2), sharey=False)
                for ax, l in zip(axes, LAYERS):
                    kinds = [("joint", "#9ca3af", -0.27), ("merged", "#ea580c" if K_ == 8 else "#2563eb", 0.0)]
                    for kind, col, dx in kinds:
                        if kind not in e: continue
                        ax.bxp([box(q, i) for i, q in enumerate(e[kind][str(l)]["q_g"])], positions=[i + dx for i in range(K_)], widths=0.24, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=col, edgecolor=col, alpha=0.85), medianprops=dict(color="black", lw=1), whiskerprops=dict(color=col), capprops=dict(color=col))
                    if "squares" in e:
                        ax.bxp([box(q, i) for i, q in enumerate(e["squares"][str(l)]["sq_q_g"])], positions=[i + 0.27 for i in range(K_)], widths=0.24, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor="white", edgecolor="#111827"), medianprops=dict(color="#111827", lw=1), whiskerprops=dict(color="#111827"), capprops=dict(color="#111827"))
                    ax.set_title(f"layer {l}" + (" (whole)" if l not in r["partitioned_layers"] else ""), fontsize=9); ax.set_xticks(range(K_)); ax.set_xticklabels([str(i) for i in range(K_)], fontsize=7)
                    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="y", color="#e5e7eb", lw=0.6); ax.set_axisbelow(True)
                axes[0].set_ylabel("router logit", fontsize=8); fig.supxlabel("square (expert group)", fontsize=8)
                fig.tight_layout(); fig.savefig(FIG / f"{model}_k{K}_{pt}.png", dpi=110); plt.close(fig); n += 1
    print(f"wrote {n} figures to {FIG}")


def main():
    res = {}; missing = []
    for model, arms in ARMS.items():
        res[model] = {}
        for K, sqn in arms.items():
            grp, part = groups_of(sqn, K); r = {"partitioned_layers": part, "experts_per_group": {str(l): int(len(grp[l][0])) for l in LAYERS}, "points": {}}
            for pt in POINTS:
                e = {}
                for kind, tag in (("start", "start"), ("joint", f"joint_{pt}"), ("merged", f"merged{K}_{pt}")):
                    if kind == "start" and pt != 19074: continue
                    if kind != "start" and pt == 19074: continue
                    z = load(model, tag)
                    if z is None: missing.append(f"{model}/{tag}"); continue
                    e[kind] = full_stats(z, grp)
                if pt != 19074:
                    zs = [load(model, f"sq{K}_g{g}_{pt}") for g in range(K)]
                    if all(z is not None for z in zs): e["squares"] = square_stats(zs, load(model, f"merged{K}_{pt}"), grp, part)
                    elif any(z is not None for z in zs): missing.append(f"{model}/sq{K}_*_{pt} (partial)")
                if e: r["points"][str(pt)] = e
            res[model][str(K)] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "router_logits.json", "w"))
    n = sum(len(p) for m in res.values() for a in m.values() for p in a["points"].values())
    print(f"wrote {OUT / 'router_logits.json'}: {n} model x point entries; missing dumps: {len(set(missing))}")
    for m in sorted(set(missing))[:40]: print("  missing", m)
    figures(res)


if __name__ == "__main__":
    main()
