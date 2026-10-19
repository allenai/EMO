#!/usr/bin/env python3
"""In-job entry point: run item ``--index`` of a ``plan.json`` written by ``launch.py``.

Paths in the plan are in job spelling (``/weka/...``); this host's short mounts are tried
too so the same command works locally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOUNTS = {"/weka/oe-training-default/ryanwang": "/root", "/weka/olmo-3p5-checkpoints": "/ckpts"}


def here(p: str) -> str:
    if os.path.exists(p):
        return p
    for job, local in MOUNTS.items():
        if p == job or p.startswith(job + "/"):
            cand = local + p[len(job):]
            if os.path.exists(cand):
                return cand
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    args, extra = ap.parse_known_args()
    plan = json.loads(Path(here(args.plan)).read_text())
    item = plan["items"][args.index]
    cmd = [sys.executable, str(HERE / "run_item.py"), "--model", here(item["model"]), "--task", item["task"], "--keep-k", str(item["keep_k"]),
           "--base-dir", here(args.base_dir), "--gpus", str(args.gpus), *extra]
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
