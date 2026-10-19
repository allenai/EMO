#!/usr/bin/env python3
"""Status of the packed selective-eval jobs: Beaker state per job, items done on disk.

    python scripts/selective_hf/olmo35/status.py [--base-dir ...] [--no-beaker] [--failed]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import MODELS  # noqa: E402
from run_item import metrics_done, relative_dir  # noqa: E402


def beaker_states(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 20):
        chunk = ids[i : i + 20]
        proc = subprocess.run(["beaker", "experiment", "get", *chunk, "--format", "json"], capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        for e in json.loads(proc.stdout):
            jobs = e.get("jobs") or []
            j = jobs[-1] if jobs else {}
            st = j.get("status", {})
            if st.get("canceled"):
                state = "canceled"
            elif st.get("exitCode") is not None:
                state = "succeeded" if st["exitCode"] == 0 else f"failed({st['exitCode']})"
            elif st.get("started"):
                state = "running"
            else:
                state = "queued"
            out[e["id"]] = {"state": state, "attempts": len(jobs), "node": j.get("node"), "started": st.get("started")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/root/scaling-ladders/output/emo_selective/selective_evals_final")
    ap.add_argument("--no-beaker", action="store_true")
    ap.add_argument("--failed", action="store_true", help="list items of failed/canceled jobs that have no metrics")
    args = ap.parse_args()
    base = Path(args.base_dir)
    jobs = []
    for rec in sorted((base / "__jobs__").glob("*/launch.json")):
        jobs.append(json.loads(rec.read_text()))
    states = {} if args.no_beaker else beaker_states([j["experiment_id"] for j in jobs if j.get("experiment_id")])
    per_model = Counter()
    per_model_total = Counter()
    job_states = Counter()
    todo = []
    for j in jobs:
        st = states.get(j.get("experiment_id"), {}).get("state", "?")
        job_states[st] += 1
        done_here = 0
        for it in j["items"]:
            model = MODELS[it["model_key"]]
            per_model_total[(it["model_key"], it["keep_k"])] += 1
            ok = metrics_done(base / relative_dir(model, it["task"], it["keep_k"]) / "results" / "checkpoint-0", it["task"])
            per_model[(it["model_key"], it["keep_k"])] += int(ok)
            done_here += int(ok)
            if not ok and (st.startswith("failed") or st == "canceled"):
                todo.append((j["name"], st, it["model_key"], it["task"], it["keep_k"]))
        if st.startswith("failed") or st == "canceled" or args.failed:
            print(f"{st:14s} {done_here:3d}/{len(j['items']):<3d} {j['name']}  {j.get('url')}")
    print("jobs:", dict(job_states))
    for key in sorted(per_model_total):
        print(f"  {key[0]:16s} k={key[1]:5s} {per_model[key]:3d}/{per_model_total[key]}")
    print(f"items done: {sum(per_model.values())}/{sum(per_model_total.values())}")
    if args.failed and todo:
        print("\nunfinished items in failed jobs:")
        for t in todo:
            print("  ", *t)


if __name__ == "__main__":
    main()
