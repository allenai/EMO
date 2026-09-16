#!/usr/bin/env python3
"""Verify that a router-only finetune changed ONLY the routed-expert routers.

Loads every `module.<name>.main` tensor (the fp32 master weights) of the start checkpoint and of the trained
checkpoint and compares them: non-router parameters must be bit-identical, router weights must differ.
Exit code 1 (and a listing of the offenders) if either condition fails.

  python check_router_only.py --start <dir with model_and_optim> --end <step dir> [--json out.json]
"""
import argparse, json, re, sys
from pathlib import Path

import torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

ROUTER = re.compile(r"\.routed_experts_router\.weight\.main$")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", type=Path, required=True); ap.add_argument("--end", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None); a = ap.parse_args()
    src = str(a.start / "model_and_optim"); dst = str(a.end / "model_and_optim")
    keys = sorted(k for k in get_checkpoint_metadata(dst).state_dict_metadata if k.startswith("module.") and k.endswith(".main"))
    src_keys = set(k for k in get_checkpoint_metadata(src).state_dict_metadata if k.endswith(".main"))
    assert set(keys) <= src_keys, f"keys in end but not in start: {sorted(set(keys) - src_keys)[:5]}"
    res = dict(n_params=len(keys), router=[], non_router_changed=[], non_router_unchanged=0, router_unchanged=[])
    for k in keys:
        x = next(load_keys(src, [k])).float(); y = next(load_keys(dst, [k])).float()
        assert x.numel() == y.numel(), (k, x.numel(), y.numel())
        diff = (x - y).abs(); same = bool(torch.equal(x, y)); mad = float(diff.max()) if diff.numel() else 0.0
        if ROUTER.search(k):
            res["router"].append(dict(key=k, max_abs_diff=mad, mean_abs_diff=float(diff.mean()), rel_change=float(diff.norm() / (x.norm() + 1e-12))))
            if same: res["router_unchanged"].append(k)
        elif same:
            res["non_router_unchanged"] += 1
        else:
            res["non_router_changed"].append(dict(key=k, max_abs_diff=mad))
    ok = not res["non_router_changed"] and not res["router_unchanged"] and len(res["router"]) > 0
    res["ok"] = ok
    print(f"{len(keys)} parameters: {len(res['router'])} router weights changed (rel. change "
          f"{min(r['rel_change'] for r in res['router']):.4f}–{max(r['rel_change'] for r in res['router']):.4f}), "
          f"{res['non_router_unchanged']} non-router parameters bit-identical, {len(res['non_router_changed'])} non-router parameters CHANGED, "
          f"{len(res['router_unchanged'])} router weights unchanged -> {'OK' if ok else 'FAIL'}")
    for r in res["non_router_changed"][:10]: print("  changed:", r)
    for k in res["router_unchanged"][:10]: print("  router unchanged:", k)
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
