#!/usr/bin/env python3
"""Submit isolated Dense-1B DR+WT+EmbedWD E1/E2/E4 frontier evaluations."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = "ai2/flex2"
POLICY = "dense_1b_dr_wt_embwd_early_frontier_v1"
RUNNER = Path("scripts/models/run_dense_1b_dr_wt_embwd_early_frontier.py")
MANIFEST = Path("scripts/models/manifests/dense-1b-dr-wt-embwd-early-frontier-v1.json")
REPORT = Path("reports/0802/data/wsd_data_loader_1b.json")
BASE_EXPERIMENTS = {128: "01M0WWNWS66NRG21QSKB87BKE7", 256: "01M0WWP30JPMSGFE91EGMT1WQP"}


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(arguments, check=True, input=input_text, capture_output=True, text=True).stdout


def load_runs(batch: int) -> list[dict[str, Any]]:
    subprocess.run(
        [sys.executable, str(RUNNER), "--manifest", str(MANIFEST), "--batch", str(batch), "--validate-only"],
        check=True,
    )
    manifest = json.loads(MANIFEST.read_text())
    return [item for item in manifest["runs"] if int(item["batchSequences"]) == batch]


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full commit hash")
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], check=True)
    upstream = command(["git", "rev-parse", "@{upstream}"]).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", revision, upstream], check=False).returncode:
        raise RuntimeError(f"revision {revision} is not reachable from pushed upstream {upstream}")


def build_spec(batch: int, revision: str, priority: str) -> dict[str, Any]:
    base = json.loads(command(["beaker", "experiment", "spec", BASE_EXPERIMENTS[batch], "--format", "json"]))
    spec = copy.deepcopy(base)
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted source experiment must have exactly one task")
    task = spec["tasks"][0]
    task["arguments"] = ["python", str(RUNNER), "--manifest", str(MANIFEST), "--batch", str(batch)]
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        variable for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (str(variable.get("name", "")).startswith("BEAKER_") and variable.get("name") != "BEAKER_TOKEN")
    ]
    for variable in task["envVars"]:
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            break
    else:
        raise RuntimeError("trusted source task has no GIT_REF")
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "hostNetworking", "synchronizedStartTimeout"):
        task.pop(key, None)
    if "minRuntime" in task["context"]:
        raise AssertionError("early-frontier task context must omit minRuntime")
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Sequential isolated 10% WSD decay plus matched heldout evaluation for Dense-1B "
        f"BS{batch} DR+WT+EmbedWD LR1e-3/WD0.3 at E1/E2/E4. One node/eight GPUs; minRuntime omitted."
    )
    return spec


def register(batch: int, runs: list[dict[str, Any]], experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    records = report.setdefault("earlyFrontierEvaluations", [])
    existing = {record.get("id") for record in records}
    if existing.intersection(item["id"] for item in runs):
        raise RuntimeError(f"BS{batch} early-frontier records already exist")
    for item in runs:
        records.append(
            {
                "id": item["id"],
                "policy": POLICY,
                "status": "submitted",
                "batchSequences": batch,
                "lr": "1e-3",
                "wd": "0.3",
                "epoch": item["epoch"],
                "sourceCheckpoint": item["sourceCheckpoint"],
                "output": item["evaluationOutput"],
                "endpointCheckpoint": item["evaluationOutput"] + f"/step{item['endpointStep']}",
                "expectedRuntimeSeconds": item["expectedRuntimeSeconds"],
                "decayFraction": 0.1,
                "experiment": experiment,
                "beaker": experiment,
                "revision": revision,
                "gpuCount": 8,
                "nodeCount": 1,
                "automaticTaskRetries": 8,
                "minRuntimeOmitted": True,
            }
        )
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text("window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(128, 256), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.batch)
    validate_revision(args.revision)
    spec = build_spec(args.batch, args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    name = f"dense-1b-dr-wt-embwd-bs{args.batch}-e1-e2-e4-frontier-v1"
    output = command(["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE], input_text=json.dumps(spec))
    print(output, end="")
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission succeeded without a parsed experiment ID")
    if args.register:
        register(args.batch, runs, identifiers[0], args.revision)


if __name__ == "__main__":
    main()
