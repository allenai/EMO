#!/usr/bin/env python3
"""Do documents that share an expert block at layer A also share one at layer B?

For one unrestricted routing pass (extract_routing.py condition dir): spectral-cluster the experts of
layer A and of layer B (k clusters each, from the token-level lift graph exactly as in
k_sweep_partition.py), assign every document (>= min tokens) to the layer-A cluster that captures the
most of its layer-A routing (purity = that share), keep the highest-purity documents per layer-A
cluster, and look at where those documents route at layer B: their layer-B purity, their layer-B
assigned cluster, and whether documents grouped together at layer A land in the same layer-B cluster
(contingency table, majority share, same-pair agreement, NMI), against a random layer-B expert
partition of the same cluster sizes and against all documents.

Usage: python layer_pair_partition.py <cond_dir> --tag emo1000_full --out <dir> --layer-a 8 --layer-b 9 --k 4 --top 1000
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location("coact", Path(__file__).resolve().parents[1] / "coactivation" / "analyze_coactivation.py")
coact = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(coact)
_spec2 = importlib.util.spec_from_file_location("ksw", Path(__file__).resolve().parent / "k_sweep_partition.py")
ksw = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(ksw)
LAYERS = list(range(1, 10))


def assign(U, lab, k):
    onehot = np.zeros((U.shape[1], k)); onehot[np.arange(U.shape[1]), lab] = 1
    mass = U @ onehot; tot = np.maximum(mass.sum(1), 1)
    return mass.argmax(1), mass.max(1) / tot, mass / tot[:, None]


def agreement(ca, cb, k):
    """contingency (k x k), per-A-group majority share, and same-pair agreement."""
    joint = np.zeros((k, k))
    for i in range(k):
        joint[i] = np.bincount(cb[ca == i], minlength=k)
    rows = np.maximum(joint.sum(1), 1)
    majority = joint.max(1) / rows                         # share of each A-group in its most common B cluster
    maj_all = float(joint.max(1).sum() / max(joint.sum(), 1))
    # same-pair agreement: P(two docs in the same B cluster | same A cluster)
    same_b_given_same_a = float(sum((joint[i] ** 2).sum() - joint[i].sum() for i in range(k)) / max(sum(rows[i] * (rows[i] - 1) for i in range(k)), 1))
    pb = joint.sum(0) / max(joint.sum(), 1); base = float((pb ** 2).sum())   # same B cluster for two random docs
    return joint, majority, maj_all, same_b_given_same_a, base, ksw.norm_mi(joint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", type=Path); ap.add_argument("--tag", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer-a", type=int, default=8); ap.add_argument("--layer-b", type=int, default=9)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--top", type=int, default=1000, help="highest-purity docs kept per layer-A cluster")
    ap.add_argument("--min-doc-tokens", type=int, default=64)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = np.load(a.cond / "counts.npz"); Ct = c["coact_tok"].astype(float); N = float(c["n_tok"])
    du = np.load(a.cond / "doc_usage.npy"); doc_len = np.load(a.cond / "doc_stats.npz")["doc_len"]
    docs = np.load(a.cond / "docs.npz", allow_pickle=True); group = docs["group"][docs["doc_inst"]]
    keep = doc_len >= a.min_doc_tokens; grp = group[keep]; groups = sorted(set(grp))
    E = Ct.shape[1]; k = a.k; rng = np.random.default_rng(0); a.out.mkdir(parents=True, exist_ok=True)
    LA, LB = LAYERS.index(a.layer_a), LAYERS.index(a.layer_b)
    UA, UB = du[keep, LA].astype(float), du[keep, LB].astype(float)
    labs = {}
    for li in (LA, LB):
        lift, _ = coact.lift_matrix(Ct[li], N); W = np.log(np.maximum(lift, 1e-9)); W[W < 0] = 0; np.fill_diagonal(W, 0)
        labs[li] = coact.spectral_labels(W, k)
    ca, pa, _ = assign(UA, labs[LA], k)
    cb, pb, fb = assign(UB, labs[LB], k)
    # selection: top-N purity documents per layer-A cluster
    sel = np.zeros(len(ca), bool)
    for i in range(k):
        idx = np.where(ca == i)[0]; idx = idx[np.argsort(-pa[idx])][: a.top]; sel[idx] = True
    # null: random layer-B expert partition with the same cluster sizes (5 draws)
    nulls = []
    for _ in range(5):
        labn = rng.permutation(labs[LB]); cbn, pbn, _ = assign(UB, labn, k)
        j, maj, maj_all, spa, base, nmi = agreement(ca[sel], cbn[sel], k)
        nulls.append(dict(purity_b=float(pbn[sel].mean()), majority=maj_all, same_pair=spa, nmi=nmi))
    null = {key: float(np.mean([n[key] for n in nulls])) for key in nulls[0]}
    res = {}
    for name, m in (("selected", sel), ("all", np.ones(len(ca), bool))):
        j, maj, maj_all, spa, base, nmi = agreement(ca[m], cb[m], k)
        res[name] = dict(n_docs=int(m.sum()), purity_a=float(pa[m].mean()), purity_b=float(pb[m].mean()),
                         purity_b_gt_half=float((pb[m] > 0.5).mean()), contingency=j.astype(int).tolist(),
                         majority_per_a_group=maj.round(3).tolist(), majority=maj_all, same_pair=spa, same_pair_base=base, nmi=nmi,
                         a_group_sizes=np.bincount(ca[m], minlength=k).tolist(), b_group_sizes=np.bincount(cb[m], minlength=k).tolist(),
                         mean_b_mass_per_a_group=np.stack([fb[m][ca[m] == i].mean(0) for i in range(k)]).round(3).tolist(),
                         source_per_a_group={g: np.bincount(ca[m][grp[m] == g], minlength=k).tolist() for g in groups})
    res["null_random_layer_b_partition_selected"] = null
    res["meta"] = dict(tag=a.tag, layer_a=a.layer_a, layer_b=a.layer_b, k=k, top_per_cluster=a.top, min_doc_tokens=a.min_doc_tokens,
                       expert_cluster_sizes_a=np.bincount(labs[LA], minlength=k).tolist(), expert_cluster_sizes_b=np.bincount(labs[LB], minlength=k).tolist(),
                       selected_purity_a_min=float(pa[sel].min()))
    json.dump(res, open(a.out / f"{a.tag}_L{a.layer_a}toL{a.layer_b}_k{k}_top{a.top}.json", "w"), indent=1)
    s, n = res["selected"], null
    print(f"{a.tag} L{a.layer_a}->L{a.layer_b} k={k}: selected {s['n_docs']} docs (purity_A >= {res['meta']['selected_purity_a_min']:.2f}, mean {s['purity_a']:.2f}); "
          f"layer-B purity {s['purity_b']:.2f} (all docs {res['all']['purity_b']:.2f}, null {n['purity_b']:.2f}); majority share {s['majority']:.2f} (null {n['majority']:.2f}); "
          f"same-pair {s['same_pair']:.2f} (base {s['same_pair_base']:.2f}, null {n['same_pair']:.2f}); NMI {s['nmi']:.2f} (null {n['nmi']:.2f})")
    print("contingency (rows = layer-A group, cols = layer-B cluster):"); print(np.array(s["contingency"]))
    # figure: contingency heatmap (row-normalised) + layer-B routing mass per A group + layer-B purity histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    J = np.array(s["contingency"], float); ax = axes[0]; im = ax.imshow(J / np.maximum(J.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for jj in range(k): ax.text(jj, i, int(J[i, jj]), ha="center", va="center", fontsize=9, color="k" if J[i, jj] / max(J[i].sum(), 1) < 0.6 else "w")
    ax.set_xlabel(f"layer-{a.layer_b} assigned cluster"); ax.set_ylabel(f"layer-{a.layer_a} group (top-{a.top} purity docs)"); ax.set_title("where each layer-A group lands at layer B"); plt.colorbar(im, ax=ax, fraction=0.046)
    M = np.array(s["mean_b_mass_per_a_group"]); ax = axes[1]; bottom = np.zeros(k)
    for jj in range(k): ax.bar(np.arange(k), M[:, jj], bottom=bottom, label=f"L{a.layer_b} cluster {jj}"); bottom += M[:, jj]
    ax.set_xticks(np.arange(k)); ax.set_xlabel(f"layer-{a.layer_a} group"); ax.set_ylabel(f"mean share of layer-{a.layer_b} routing"); ax.legend(fontsize=7); ax.set_title(f"layer-{a.layer_b} routing mass by layer-{a.layer_a} group")
    ax = axes[2]; ax.hist(pb[sel], bins=40, range=(0, 1), alpha=0.7, label=f"selected docs ({s['n_docs']})"); ax.hist(pb, bins=40, range=(0, 1), alpha=0.4, label="all docs", density=False)
    ax.set_xlabel(f"layer-{a.layer_b} purity"); ax.legend(fontsize=8); ax.set_title(f"layer-{a.layer_b} purity")
    fig.suptitle(f"{a.tag}: layer {a.layer_a} expert blocks (k={k}) -> layer {a.layer_b}", fontsize=11); fig.tight_layout()
    coact.save_png(fig, a.out / f"{a.tag}_L{a.layer_a}toL{a.layer_b}_k{k}_top{a.top}.png", 100); plt.close(fig)


if __name__ == "__main__":
    main()
