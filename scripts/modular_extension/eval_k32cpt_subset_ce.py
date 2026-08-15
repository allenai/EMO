#!/usr/bin/env python3
"""Held-out CE of per-cluster 32-expert SUBSETS sliced from a pool state (one GPU).

For each target cluster c: slice the (32 standard + 1 shared)-expert model using cluster
c's frozen per-layer selection (expert_concentration.json, same semantics as
expert_subset_surgery.extract) from a bf16 pool snapshot (or an anchor .pt), build the
33-expert model (routing renormalizes over the 32 standard experts -- the training-time
condition), and evaluate on cluster c's held-out stream. Complements the full-64-expert
grid: does the cluster's own working set forget as much as the full pool does?

Usage (one job shards target clusters across GPUs):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:src python scripts/modular_extension/eval_k32cpt_subset_ce.py \\
      --snapshot <pool_after_cXX.pt> --config-from <pool_dir> \\
      --selection-json <expert_concentration.json> \\
      --targets 0-7 --tokens-root <...> --out <...>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from eval_k32cpt_ce import SEQ_LEN, eval_cluster, parse_clusters  # noqa: E402

E_POOL, NS_POOL, E_SUB, NS_SUB, N_LAYERS = 64, 63, 33, 32, 16
EXPERT_SUFFIXES = ("feed_forward_moe.experts.mlp.w1", "feed_forward_moe.experts.mlp.w2",
                   "feed_forward_moe.experts.mlp.w3")
ROUTER_SUFFIX = "feed_forward_moe.router.weight"


def slice_slots(t, slot_ids):
    per = t.shape[0] // E_POOL
    v = t.reshape(E_POOL, per, *t.shape[1:])
    return v[torch.tensor(slot_ids)].reshape(len(slot_ids) * per, *t.shape[1:])


def subset_state(pool_sd, slots_per_layer):
    out = {}
    for k, v in pool_sd.items():
        if any(k.endswith(s) for s in EXPERT_SUFFIXES) or k.endswith(ROUTER_SUFFIX):
            layer = int(k.split(".")[1])
            slot_ids = slots_per_layer[str(layer)] + [NS_POOL]  # shared last
            out[k] = slice_slots(v, slot_ids)
        else:
            out[k] = v
    return out


def build_subset_model(experiment_config, device):
    from olmo_core.nn.transformer import TransformerConfig

    tcfg = json.loads(json.dumps(experiment_config["model"]))  # deep copy
    moe = tcfg["block"]["feed_forward_moe"]
    moe["num_experts"] = E_SUB
    router = moe.get("router") or {}
    for key in ("max_document_expert_pool", "eval_document_expert_pool"):
        if key in router:
            router[key] = E_SUB
    model = TransformerConfig.from_dict(tcfg).build(init_device="meta")
    model.to_empty(device=device)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, help="bf16 pool state .pt (flat fqn dict)")
    p.add_argument("--config-from", required=True, help="dir with the 64-expert config.json")
    p.add_argument("--selection-json", required=True, help="expert_concentration.json")
    p.add_argument("--targets", default="0-31", help="target clusters (each evaluated on its own heldout)")
    p.add_argument("--tokens-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens-per-cluster", type=int, default=25_000_000)
    args = p.parse_args()

    device = torch.device("cuda")
    experiment_config = json.load(open(Path(args.config_from) / "config.json"))
    conc = json.load(open(args.selection_json))
    sel_by_cluster = {c["cluster"]: {str(layer): sorted(ids)
                                     for layer, ids in enumerate(c["top32_experts_per_layer"])}
                      for c in conc["clusters"]}
    pool_sd = torch.load(args.snapshot, map_location="cpu")

    results = {}
    for c in parse_clusters(args.targets):
        t0 = time.time()
        model = build_subset_model(experiment_config, device)
        sd = {k: v.to(torch.float32) for k, v in subset_state(pool_sd, sel_by_cluster[c]).items()}
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        model.eval()
        hp = Path(args.tokens_root) / f"cluster{c:02d}" / "heldout.npy"
        ce, n, dt = eval_cluster(model, hp, device, args.batch_size, args.max_tokens_per_cluster)
        results[str(c)] = {"ce": round(ce, 5), "positions": n, "seconds": round(dt, 1)}
        del model
        torch.cuda.empty_cache()
        print(f"target {c:2d}: subset CE {ce:.4f} over {n / 1e6:.1f}M positions "
              f"({time.time() - t0:.0f}s incl build)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"snapshot": args.snapshot, "results": results}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
