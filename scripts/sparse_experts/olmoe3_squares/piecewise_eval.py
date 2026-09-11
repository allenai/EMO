#!/usr/bin/env python3
"""Piecewise held-out evaluation of the olmoe3_squares sub-models.

Assign every held-out document to a group by the FULL start model's unrestricted routing (the group
receiving most of its layer 2-9 selections, from runs_heldout20b/emo_step19074/none/doc_usage.npy),
then score each document with ITS OWN sub-model's per-document CE (runs_heldout20b/sub{g}_<name>/none).
"piecewise CE" = token-weighted CE when every document is served by its own square with no merging;
compare with the merged model's oracle-routed CE (same documents, same group choice up to the
merged model's own routing) and its plain CE. Also reports every sub-model on every group.

Usage: python piecewise_eval.py --name match25000 [--groups groups.json]
"""
import argparse, json
from pathlib import Path
import numpy as np

H = Path("sparse_experts/olmoe3_routing/runs_heldout20b")


def ce_of(d):
    s = np.load(d / "doc_stats.npz"); return s["ce_sum"], s["ce_len"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--groups", type=Path, default=Path("sparse_experts/olmoe3_squares/groups.json"))
    a = ap.parse_args(); G = json.load(open(a.groups)); k = G["k"]; PL = [int(l) for l in G["partitioned_layers"]]
    u = np.load(H / "emo_step19074/none/doc_usage.npy", mmap_mode="r")
    mass = np.zeros((u.shape[0], k))
    for l in PL:
        oh = np.zeros((G["num_experts"], k))
        for g in range(k): oh[G["groups"][str(l)][g], g] = 1
        mass += np.asarray(u[:, l - 1, :]).astype(float) @ oh
    grp = mass.argmax(1)
    subs = {g: ce_of(H / f"sub{g}_{a.name}/none") for g in range(k) if (H / f"sub{g}_{a.name}/none/doc_stats.npz").exists()}
    out = dict(name=a.name, docs_per_group=np.bincount(grp, minlength=k).tolist(), sub_on_group={}, piecewise=None)
    tot_sum = tot_len = 0.0
    for g, (cs, cl) in subs.items():
        out["sub_on_group"][str(g)] = {str(h): float(cs[grp == h].sum() / max(cl[grp == h].sum(), 1)) for h in range(k)}
        out["sub_on_group"][str(g)]["all"] = float(cs.sum() / cl.sum())
        tot_sum += cs[grp == g].sum(); tot_len += cl[grp == g].sum()
    if len(subs) == k: out["piecewise"] = float(tot_sum / tot_len)
    for tag, d in (("start_full", "emo_step19074/none"), ("start_oracle", "emo_step19074/oracle"), (f"merged_{a.name}", f"merged_{a.name}/none"), (f"merged_{a.name}_oracle", f"merged_{a.name}/oracle")):
        p = H / d / "doc_stats.npz"
        if p.exists():
            cs, cl = ce_of(H / d); out[tag] = float(cs.sum() / cl.sum()); out[tag + "_by_group"] = {str(h): float(cs[grp == h].sum() / cl[grp == h].sum()) for h in range(k)}
    Path("claude_outputs/olmoe3_routing/squares").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"claude_outputs/olmoe3_routing/squares/piecewise_{a.name}.json", "w"), indent=1)
    print(json.dumps({key: out[key] for key in out if not key.endswith("_by_group")}, indent=1))


if __name__ == "__main__":
    main()
