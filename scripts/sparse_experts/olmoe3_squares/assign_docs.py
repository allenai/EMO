#!/usr/bin/env python3
"""Stage 1 of olmoe3_squares: route the next 10B tokens through the full EMO 512e model and assign
every document to one of the k block-groups.

For each training-stream instance (sparse_experts/olmoe3_routing/stream_10b_20b, training order),
run the unrestricted full model, take the top-16 experts every token selected in each MoE layer,
and count per document how many of those selections fall into each block-group's experts
(layers 2..9; layer 1 is not partitioned). The document goes to the group with the most
selections. Saved per document: instance id, token span, chosen group, the (layer, group) count
table (the "natural routing" and what a sub-model would miss), and the full model's mean CE.
Shards are split round-robin over --world ranks; each rank writes records_<rank>.npz.

Usage (one GPU): python assign_docs.py --checkpoint <emo step19074> --stream <stream dir> \
    --groups sparse_experts/olmoe3_squares/groups.json --out <dir> --rank r --world w
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "olmoe3_routing"))
import extract_routing as ex  # build_model, doc_table, MOE_LAYERS, EOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--stream", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--max-instances", type=int, default=0, help="per shard, for smoke tests")
    ap.add_argument("--attn-backend", default=None); ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args()
    from olmo_core.data.utils import get_labels
    device = torch.device("cuda")
    G = json.load(open(a.groups)); k = G["k"]; E = G["num_experts"]; PL = [int(l) for l in G["partitioned_layers"]]
    man = json.load(open(a.stream / "manifest.json")); shards = [s["k"] for s in man["shards"]]
    if a.max_shards: shards = shards[: a.max_shards]
    my = shards[a.rank :: a.world]
    n_per_shard = {s["k"]: s["n_instances"] for s in man["shards"]}
    base = {}; off = 0
    for s in man["shards"]: base[s["k"]] = off; off += s["n_instances"]   # global instance index = training order
    # expert -> group lookup per partitioned layer
    e2g = torch.full((len(ex.MOE_LAYERS), E), -1, dtype=torch.long)
    for l in PL:
        for g in range(k): e2g[ex.MOE_LAYERS.index(l), G["groups"][str(l)][g]] = g
    e2g = e2g.to(device); pl_idx = torch.tensor([ex.MOE_LAYERS.index(l) for l in PL], device=device)
    # size-normalised assignment (groups.json flag): argmax of mass per expert of the group, so uneven blocks
    # with token-level routing do not all collapse onto the biggest block
    n_exp = torch.tensor([sum(len(G["groups"][str(l)][g]) for l in PL) for g in range(k)], device=device, dtype=torch.float)
    norm = n_exp if G.get("size_normalized") else torch.ones_like(n_exp)
    ex.log(f"assignment rule: {'mass per expert' if G.get('size_normalized') else 'raw mass'}; group expert totals {n_exp.tolist()}")
    model, routers, mcfg = ex.build_model(a.checkpoint, device, a.attn_backend, False)
    topk = mcfg["block"]["routed_experts_router"]["top_k"]
    captured = {}
    def make_hook(l):
        def hook(mod, inp, out): captured[l] = out[1]
        return hook
    hooks = [routers[l].register_forward_hook(make_hook(l)) for l in ex.MOE_LAYERS]
    a.out.mkdir(parents=True, exist_ok=True)
    recs = dict(inst=[], start=[], end=[], group=[], share=[], ce=[], mass=[], label=[])
    t0 = time.time(); done_tok = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for sk in my:
            tokens = np.load(a.stream / f"shard_{sk}.tokens.npy", mmap_mode="r"); meta = np.load(a.stream / f"shard_{sk}.meta.npz")
            N = n_per_shard[sk] if not a.max_instances else min(a.max_instances, n_per_shard[sk]); S = tokens.shape[1]
            for b0 in range(0, N, a.batch_size):
                tb = np.asarray(tokens[b0 : b0 + a.batch_size]); B = tb.shape[0]
                doc_of, n_doc, doc_inst = ex.doc_table(tb)
                ids = torch.from_numpy(tb.astype(np.int64)).to(device); labels = get_labels({"input_ids": ids})
                out = model(ids, labels=labels, loss_reduction="none", return_logits=False)
                loss = out.ce_loss if hasattr(out, "ce_loss") else out
                idx = torch.stack([captured[l] for l in ex.MOE_LAYERS], dim=2).long()   # (B, S, L, k)
                doc = torch.from_numpy(doc_of).to(device)                                  # (B, S)
                # group of every selection in the partitioned layers -> per-doc (layer, group) counts
                gsel = torch.gather(e2g[pl_idx][None, None].expand(B, S, -1, -1), 3, idx[:, :, pl_idx])  # (B,S,PL,k)
                flat_doc = doc.reshape(-1)                                                     # (B*S,)
                onehot = torch.nn.functional.one_hot(gsel.reshape(B * S, len(PL), topk), k).sum(2)  # (B*S, PL, k)
                mass = torch.zeros(n_doc, len(PL), k, device=device, dtype=torch.long).index_add_(0, flat_doc, onehot)
                tot = mass.sum(1)                                                              # (n_doc, k)
                grp = (tot.float() / norm).argmax(1); share = tot.gather(1, grp[:, None])[:, 0].float() / tot.sum(1).clamp(min=1).float()
                # doc CE: loss at t predicts t+1 -> attribute to doc of t+1
                loss_doc = torch.roll(doc, -1, 1).reshape(-1); valid = (labels != -100).reshape(-1)
                ce_sum = torch.zeros(n_doc, device=device).index_add_(0, loss_doc[valid], loss.reshape(-1)[valid].float())
                ce_n = torch.zeros(n_doc, device=device).index_add_(0, loss_doc[valid], torch.ones(int(valid.sum()), device=device))
                # spans
                d_np = doc_of; starts = np.zeros(n_doc, np.int32); ends = np.zeros(n_doc, np.int32)
                for i in range(B):
                    row = d_np[i]; change = np.flatnonzero(np.diff(row)) + 1
                    s_pos = np.concatenate([[0], change]); e_pos = np.concatenate([change, [S]])
                    ds = row[s_pos]; starts[ds] = s_pos; ends[ds] = e_pos
                recs["inst"].append(base[sk] + b0 + doc_inst); recs["start"].append(starts); recs["end"].append(ends)
                recs["group"].append(grp.cpu().numpy().astype(np.int8)); recs["share"].append(share.cpu().numpy().astype(np.float16))
                recs["ce"].append((ce_sum / ce_n.clamp(min=1)).cpu().numpy().astype(np.float16)); recs["mass"].append(mass.cpu().numpy().astype(np.int32))
                recs["label"].append(meta["label_id"][b0 : b0 + B][doc_inst].astype(np.int16))
                done_tok += B * S
                if b0 == 0 and sk == my[0]:
                    ce0 = loss[labels != -100].mean().item(); assert ce0 < 4.0, f"first-batch CE {ce0:.3f}: weights mis-loaded?"
                if (b0 // a.batch_size) % a.log_every == 0:
                    el = time.time() - t0; ex.log(f"shard {sk} {b0+B}/{N} inst, {done_tok/el/1e3:.0f}k tok/s, CE {loss[labels != -100].mean().item():.3f}")
    for h in hooks: h.remove()
    np.savez(a.out / f"records_{a.rank}.npz", **{key: np.concatenate(v) for key, v in recs.items()})
    (a.out / f"DONE_{a.rank}").write_text("ok")
    ex.log(f"rank {a.rank}: wrote {sum(len(x) for x in recs['group'])} docs from shards {my}")


if __name__ == "__main__":
    main()
