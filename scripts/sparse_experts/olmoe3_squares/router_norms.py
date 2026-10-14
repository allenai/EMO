#!/usr/bin/env python3
"""Router-weight norms of the random-split squares vs the jointly trained baseline (user request 2026-10-14, Q5 box 5A).
For an arm (K squares at sub-model LR 4e-4, window-2 finals = the 30B point) and every MoE layer: the routed-expert router rows
(one d_model vector per expert, no bias) of each square are compared with the SAME experts' rows in the joint baseline's 30B
checkpoint and in the 10B start model: per-row L2 norms (mean, std), the norm of the group's mean row (a direction shared by a
group's rows adds one logit offset to all of its experts after the merge, which is invisible inside the square's own softmax),
the mean cosine to the start rows and the relative change ||W - S|| / ||S||. Layer 1 is whole in every square (all E rows).
  python scripts/sparse_experts/olmoe3_squares/router_norms.py --out claude_outputs/olmoe3_routing/router_norms.json
"""
import argparse, json, re
from pathlib import Path
import torch

S = Path("sparse_experts"); D_MODEL = 640; LAYERS = list(range(1, 10))
ARMS = {  # arm -> (groups.json dir, square run prefix, K)
    "k8_lr4e-4": ("olmoe3_squares_emorand8_lr4e-4", "olmoe3_275m_emorand8_lr4e-4_square", 8),
    "k4_lr4e-4": ("olmoe3_squares_emorand_lr4e-4", "olmoe3_275m_emorand_lr4e-4_square", 4),
}
BASE = S / "olmoe3_275m_emo_30b_1node/step57221/model_and_optim"; START = S / "olmoe3_275m_emo_10b/step19074/model_and_optim"


def router(dcp_dir, layer, n_rows):
    from olmo_core.distributed.checkpoint import load_keys
    key = f"module.blocks.{layer}.routed_experts_router.weight.main"
    return next(load_keys(str(dcp_dir), [key])).float().reshape(n_rows, D_MODEL)


def final_step(run):
    steps = sorted(int(m.group(1)) for d in (S / run).glob("step*") if (m := re.fullmatch(r"step(\d+)", d.name)))
    return steps[-1]


def stats(W, B, St):
    nw, nb, ns = W.norm(dim=1), B.norm(dim=1), St.norm(dim=1)
    cos = lambda A, C: torch.nn.functional.cosine_similarity(A, C, dim=1).mean().item()
    rel = lambda A, C: ((A - C).norm() / C.norm()).item()
    return dict(n=int(W.shape[0]), norm_sq=nw.mean().item(), norm_sq_std=nw.std().item(), norm_base=nb.mean().item(), norm_base_std=nb.std().item(), norm_start=ns.mean().item(),
                centroid_sq=W.mean(0).norm().item(), centroid_base=B.mean(0).norm().item(), centroid_start=St.mean(0).norm().item(),
                cos_sq_start=cos(W, St), cos_base_start=cos(B, St), cos_sq_base=cos(W, B), rel_sq_start=rel(W, St), rel_base_start=rel(B, St),
                # the group's mean logit offset relative to the baseline for a typical unit input along the centroid: (W - B) mean row norm
                centroid_shift=(W.mean(0) - B.mean(0)).norm().item())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    out = {}
    for arm, (sqn, rp, K) in ARMS.items():
        G = json.load(open(S / sqn / "groups.json")); E = G["num_experts"]; res = {}
        subs = {g: S / f"{rp}{g}_w2" / f"step{final_step(f'{rp}{g}_w2')}" / "model_and_optim" for g in range(K)}
        for l in LAYERS:
            B = router(BASE, l, E); St = router(START, l, E); res[l] = {}
            for g in range(K):
                ids = torch.tensor(G["groups"][str(l)][g]); W = router(subs[g], l, len(ids))
                res[l][g] = stats(W, B[ids], St[ids])
            res[l]["all_base"] = dict(norm=B.norm(dim=1).mean().item(), centroid=B.mean(0).norm().item()); res[l]["all_start"] = dict(norm=St.norm(dim=1).mean().item(), centroid=St.mean(0).norm().item())
            # dispersion of the group centroids: mean pairwise distance between the groups' mean rows (squares' rows vs the same experts in the baseline / start):
            # after the merge, a difference between two groups' shared directions is a logit offset between those groups
            cents = {name: torch.stack([M[torch.tensor(G["groups"][str(l)][g])].mean(0) if name != "sq" else router(subs[g], l, len(G["groups"][str(l)][g])).mean(0) for g in range(K)]) for name, M in (("sq", None), ("base", B), ("start", St))}
            pd = lambda C: torch.pdist(C).mean().item()
            res[l]["centroid_pdist"] = dict(sq=pd(cents["sq"]), base=pd(cents["base"]), start=pd(cents["start"]))
            print(f"{arm} layer {l}: " + " ".join(f"g{g} {res[l][g]['norm_sq']:.3f}/{res[l][g]['norm_base']:.3f}" for g in range(K)) + f" (start {res[l]['all_start']['norm']:.3f})", flush=True)
        out[arm] = dict(K=K, E=E, sqn=sqn, squares={g: str(subs[g]) for g in range(K)}, baseline=str(BASE), start=str(START), layers={str(l): {str(k): v for k, v in res[l].items()} for l in LAYERS})
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
