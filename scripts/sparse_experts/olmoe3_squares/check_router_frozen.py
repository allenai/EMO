#!/usr/bin/env python3
"""Sanity check for the frozen-router squares (user request 2026-09-24): a merged model's routed-expert routers (every MoE layer)
must be bit-identical to the start model's, and its experts must have changed. Compares the fp32 master weights
(`module.<name>.main`) tensor by tensor; exit 1 with a listing if either condition fails.
  python check_router_frozen.py --start <step dir of the start model> --merged <merged dir> [--json out.json]
"""
import argparse, json, sys
from pathlib import Path
import torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", type=Path, required=True); ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None); a = ap.parse_args()
    src = str(a.start / "model_and_optim"); dst = str(a.merged / "model_and_optim")
    keys = sorted(k for k in get_checkpoint_metadata(dst).state_dict_metadata if k.startswith("module.") and k.endswith(".main"))
    routers = [k for k in keys if ".routed_experts_router.weight." in k]; experts = [k for k in keys if ".routed_experts.w_" in k]
    bad_router, same_expert, max_abs = [], [], 0.0
    for k in routers:
        s = next(load_keys(src, [k])); m = next(load_keys(dst, [k]))
        if not torch.equal(s, m): bad_router.append(k); max_abs = max(max_abs, float((s.float() - m.float()).abs().max()))
    for k in experts:
        s = next(load_keys(src, [k])); m = next(load_keys(dst, [k]))
        if torch.equal(s, m): same_expert.append(k)
    ok = not bad_router and not same_expert
    res = dict(ok=ok, n_router_tensors=len(routers), routers_changed=bad_router, max_abs_router_diff=max_abs, n_expert_tensors=len(experts), experts_unchanged=same_expert)
    if a.json: a.json.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.json, "w"), indent=1)
    print(("OK" if ok else "FAILED") + f": {len(routers)} router tensors identical to the start model" + (f" except {bad_router}" if bad_router else "")
          + f"; {len(experts) - len(same_expert)}/{len(experts)} expert tensors changed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
