#!/usr/bin/env python3
"""Sanity check for the freeze ablations (user request 2026-10-08): in a merged (or square) checkpoint, every fp32 master tensor whose
name matches one of the frozen globs must be bit-identical to the start model's, and every other tensor must have changed.
Globs are the OLMOE3_FREEZE_PATTERNS ones (fnmatch against "module.<name>"). Exit 1 with a listing if either condition fails.
  python check_frozen.py --start <step dir> --model <dir> --frozen "<glob>,<glob>" [--json out.json]
Layer-1 routed experts / routers live in every square (layer 1 is whole), so a merged model's layer-1 tensors are checked like the rest.
"""
import argparse, fnmatch, json, sys
from pathlib import Path
import torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", type=Path, required=True); ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--frozen", required=True); ap.add_argument("--json", type=Path, default=None); a = ap.parse_args()
    pats = [p for p in a.frozen.split(",") if p]
    src = str(a.start / "model_and_optim"); dst = str(a.model / "model_and_optim")
    keys = sorted(k for k in get_checkpoint_metadata(dst).state_dict_metadata if k.startswith("module.") and k.endswith(".main"))
    frozen = [k for k in keys if any(fnmatch.fnmatch(k[:-len(".main")], p) for p in pats)]; free = [k for k in keys if k not in set(frozen)]
    changed_frozen, same_free, maxd = [], [], 0.0
    for k in frozen:
        s = next(load_keys(src, [k])); m = next(load_keys(dst, [k]))
        if not torch.equal(s, m): changed_frozen.append(k); maxd = max(maxd, float((s.float() - m.float()).abs().max()))
    for k in free:
        s = next(load_keys(src, [k])); m = next(load_keys(dst, [k]))
        if torch.equal(s, m): same_free.append(k)
    ok = not changed_frozen and not same_free
    res = dict(ok=ok, patterns=pats, n_frozen=len(frozen), n_trainable=len(free), frozen_changed=changed_frozen, max_abs_frozen_diff=maxd, trainable_unchanged=same_free)
    if a.json: a.json.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.json, "w"), indent=1)
    print(("OK" if ok else "FAILED") + f": {len(frozen)} frozen tensors identical to the start model" + (f" except {len(changed_frozen)}: {changed_frozen[:4]}" if changed_frozen else "")
          + f"; {len(free) - len(same_free)}/{len(free)} trainable tensors changed" + (f" (unchanged: {same_free[:4]})" if same_free else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
