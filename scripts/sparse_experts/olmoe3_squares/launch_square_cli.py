#!/usr/bin/env python3
"""Launch the squares of a NEW arm through the Beaker CLI from the specs of a finished reference arm (no olmo_core import).

Why: on a cold host the first import of the pinned OLMo-core takes from many minutes to hours (the env lives on weka), and every
Python launcher hangs until it is warm. A square job reads its whole configuration from OLMOE3_* environment variables, so a new arm
that shares the reference arm's expert groups, document packs and start slices is the reference square's spec with the arm names
replaced, GIT_REF moved to the pushed HEAD and a few variables added. Stdlib only.

For each group g: `beaker experiment spec <reference square g>` -> rename the arm (--ref-sqn -> --sqn, --ref-rp -> --rp), set GIT_REF,
add --env KEY=VALUE, minRuntime 8h -> `beaker experiment create` -> marker <sqn>/logs/square<g>[_w2]_launched (the same marker the
squares_lr_sweep.sh driver writes, so the driver then only waits for the checkpoints, merges and evaluates). The arm directory
(groups.json copy, pack / pack2 / init symlinks to the parent arm) is created if missing.

  python scripts/sparse_experts/olmoe3_squares/launch_square_cli.py --ref-sqn olmoe3_squares_emorand8_lr4e-4 --ref-rp olmoe3_275m_emorand8_lr4e-4_square \
      --sqn olmoe3_squares_emorand8_lr4e-4_top4 --rp olmoe3_275m_emorand8_lr4e-4_top4_square --parent olmoe3_squares_emorand8 \
      --groups 0 --window 1 --env OLMOE3_SQUARE_TOP_K=4 --tag square_top4 [--dry-run]        (commit + push first)
"""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; S = ROOT / "sparse_experts"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-sqn", required=True); ap.add_argument("--ref-rp", required=True); ap.add_argument("--sqn", required=True); ap.add_argument("--rp", required=True)
    ap.add_argument("--parent", required=True, help="arm that owns pack / pack2 / init (symlinked into the new arm)")
    ap.add_argument("--groups", required=True, help="comma-separated group indices"); ap.add_argument("--window", type=int, default=1, choices=[1, 2])
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the worker"); ap.add_argument("--tag", default="", help="extra W&B tag")
    ap.add_argument("--workspace", default="ai2/flex2"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/ryanyxw/emo_extend"], text=True).split()[0]
    if head != remote: sys.exit(f"HEAD {head[:9]} is not pushed (origin emo_extend = {remote[:9]}): the workers clone GIT_REF")
    sq = S / a.sqn; logs = sq / "logs"; logs.mkdir(parents=True, exist_ok=True)
    if not (sq / "groups.json").exists(): shutil.copy(S / a.ref_sqn / "groups.json", sq / "groups.json")
    for d in ("pack", "pack2", "init"):
        if not (sq / d).exists() and not (sq / d).is_symlink(): os.symlink(f"../{a.parent}/{d}", sq / d)
    suffix = "" if a.window == 1 else "_w2"
    for g in [int(x) for x in a.groups.split(",")]:
        mark = logs / f"square{g}{suffix}_launched"
        if mark.exists(): print(f"group {g}: marker exists ({mark.read_text().strip()}), skipped"); continue
        if a.window == 1 and not (sq / "init" / f"group{g}" / ".metadata").exists(): sys.exit(f"start slice {sq}/init/group{g} missing")
        if a.window == 2 and not (sq / "init2" / f"group{g}" / "model_and_optim" / ".metadata").exists(): sys.exit(f"window-2 start {sq}/init2/group{g} missing")
        ref = (S / a.ref_sqn / "logs" / f"square{g}{suffix}_launched").read_text().strip().split("/")[-1]
        rc, txt = sh("beaker", "experiment", "spec", ref, "--format", "json")
        if rc: sys.exit(f"spec of {ref} failed: {txt[-300:]}")
        spec = json.loads(txt[txt.index("{"):]); raw = json.dumps(spec)
        raw = raw.replace(a.ref_rp, a.rp).replace(a.ref_sqn, a.sqn)       # run name (arguments) and every arm path / tag
        spec = json.loads(raw); (task,) = spec["tasks"]
        run = task["arguments"][3]; assert run == f"{a.rp}{g}{suffix}", run
        env = {e["name"]: e for e in task["envVars"]}
        env["GIT_REF"]["value"] = head
        for kv in a.env:
            k, v = kv.split("=", 1); env[k] = {"name": k, "value": v}
        if a.tag: env["OLMOE3_WANDB_TAGS"]["value"] += "," + a.tag
        task["envVars"] = list(env.values()); task["context"]["minRuntime"] = "8h"
        spec["description"] = f"{run} (sparser squares; spec of {ref} at {head[:9]})"
        show = {k: env[k].get("value") for k in sorted(env) if k.startswith("OLMOE3_") or k == "GIT_REF"}
        if a.dry_run: print(f"group {g}: would create {run}\n  " + "\n  ".join(f"{k}={v}" for k, v in show.items())); continue
        f = logs / f"spec_square{g}{suffix}.json"; f.write_text(json.dumps(spec, indent=1))
        rc, txt = sh("beaker", "experiment", "create", str(f), "--name", run, "--workspace", a.workspace)
        m = re.search(r"beaker\.org/ex/[A-Z0-9]+", txt) or re.search(r"\b(01[A-Z0-9]{24})\b", txt)
        if rc or not m: sys.exit(f"create {run} failed: {txt[-400:]}")
        url = m.group(0) if m.group(0).startswith("beaker.org") else f"beaker.org/ex/{m.group(1)}"
        mark.write_text(url + "\n"); print(f"group {g}: {run} -> {url}", flush=True)


if __name__ == "__main__":
    main()
