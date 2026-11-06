#!/usr/bin/env python3
"""olmoe3_beaker_cmd.py without the olmo_core import: create an allocated 1-GPU jupiter job that runs one command at the pushed HEAD, from
the spec of a finished 1-GPU job (`beaker experiment spec` -> arguments, GIT_REF, minRuntime replaced -> `beaker experiment create`).
For cold hosts, where the first olmo_core import takes minutes to hours (see launch_square_cli.py). Prints beaker.org/ex/<id> like the
Python launcher, so the drivers' URL grep works unchanged. Stdlib only.
  python scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name <job name> [--gpus 1] -- python scripts/... (or bash -c "...")
"""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = "01M6F20487GRQHZ19DS2NT8JYM"   # lbloss-emo2000-merged8_step38148 (2026-11-02): allocated, 1 GPU, ai2/jupiter, urgent, PYTHONPATH external/OLMo-core/src:src


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--ref", default=REF)
    ap.add_argument("--workspace", default="ai2/flex2"); ap.add_argument("--allocated", action="store_true", help="accepted for compatibility (the reference job is allocated)")
    ap.add_argument("--unallocated", action="store_true", help="unallocated filler slot (minRuntime 0, autoResume): starts in minutes when the allocated 1-GPU queue is stuck, may be preempted")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("cmd", nargs=argparse.REMAINDER); a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd: sys.exit("no command")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/ryanyxw/emo_extend"], text=True).split()[0]
    if head != remote: sys.exit(f"HEAD {head[:9]} is not pushed (origin emo_extend = {remote[:9]})")
    r = subprocess.run(["beaker", "experiment", "spec", a.ref, "--format", "json"], capture_output=True, text=True)
    if r.returncode: sys.exit(f"spec of {a.ref} failed: {(r.stdout + r.stderr)[-300:]}")
    spec = json.loads(r.stdout[r.stdout.index("{"):]); (task,) = spec["tasks"]
    task["arguments"] = cmd; task["resources"]["gpuCount"] = a.gpus; task["context"]["minRuntime"] = "8h"
    if a.unallocated: task["context"]["minRuntime"] = "0s"; task["context"]["autoResume"] = True
    for e in task["envVars"]:
        if e["name"] == "GIT_REF": e["value"] = head
    spec["description"] = f"{a.name} ({head[:9]})"
    if a.dry_run: print(json.dumps(task["arguments"])[:600]); return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(spec, f); path = f.name
    r = subprocess.run(["beaker", "experiment", "create", path, "--name", a.name, "--workspace", a.workspace], capture_output=True, text=True)
    txt = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr); m = re.search(r"beaker\.org/ex/[A-Z0-9]+", txt) or re.search(r"\b(01[A-Z0-9]{24})\b", txt)
    if r.returncode or not m: sys.exit(f"create {a.name} failed: {txt[-400:]}")
    print(m.group(0) if m.group(0).startswith("beaker.org") else f"https://beaker.org/ex/{m.group(1)}")


if __name__ == "__main__":
    main()
