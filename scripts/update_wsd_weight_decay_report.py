#!/usr/bin/env python3
"""Upsert one run in the data-driven WSD weight-decay report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "reports/data/wsd_weight_decay_1b.json"
JS_PATH = ROOT / "reports/data/wsd_weight_decay_1b.js"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--series", choices=("wd", "unique"), default="wd")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--lr", required=True)
    p.add_argument("--wd", required=True)
    p.add_argument("--status", choices=("queued", "active", "complete", "canceled", "failed"), required=True)
    p.add_argument("--train", type=float)
    p.add_argument("--c4", type=float)
    p.add_argument("--acc", type=float)
    p.add_argument("--bpb", type=float)
    p.add_argument("--wandb")
    p.add_argument("--beaker")
    return p


def main() -> None:
    args = parser().parse_args()
    state = json.loads(JSON_PATH.read_text())
    key = (args.epoch, args.lr, args.wd)
    runs_key = "runs" if args.series == "wd" else "uniqueRuns"
    state.setdefault(runs_key, [])
    run = next(
        (r for r in state[runs_key] if (r["epoch"], r["lr"], r["wd"]) == key),
        None,
    )
    if run is None:
        run = {"epoch": args.epoch, "lr": args.lr, "wd": args.wd}
        state[runs_key].append(run)
    for name in ("status", "train", "c4", "acc", "bpb", "wandb", "beaker"):
        value = getattr(args, name)
        if value is not None:
            run[name] = value
    state[runs_key].sort(key=lambda r: (r["epoch"], float(r["wd"]), float(r["lr"])))
    state["updated"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
    rendered = json.dumps(state, indent=2, sort_keys=False) + "\n"
    JSON_PATH.write_text(rendered)
    JS_PATH.write_text("window.WD_REPORT_DATA = " + json.dumps(state, separators=(",", ":")) + ";\n")
    print(f"updated series={args.series} epoch={args.epoch} lr={args.lr} wd={args.wd} status={args.status}")


if __name__ == "__main__":
    main()
