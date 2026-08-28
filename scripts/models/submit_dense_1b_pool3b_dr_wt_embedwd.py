#!/usr/bin/env python3
"""Submit the two isolated Dense-1B Pool-3B DR+WT+EmbedWD chains."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = "ai2/flex2"
BASE_EXPERIMENT = "01M0WWNWS66NRG21QSKB87BKE7"
POLICY = "dense_1b_pool3b_dr_wt_embedwd_postdecay_saturation_v1"
REPORT = Path("reports/0802/data/wsd_data_loader_1b_pool3b_drwtembwd.json")
REPORT_JS = REPORT.with_suffix(".js")
RUNNER = "scripts/models/run_dense_1b_pool3b_dr_wt_embedwd.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
    ).stdout


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_POOL3B_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == name:
            variable["value"] = value
            return
    task.setdefault("envVars", []).append({"name": name, "value": value})


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("--revision must be a full 40-character commit hash")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"],
        check=True,
    )


def experiment_name(batch: int) -> str:
    return f"dense-1b-pool3b-bs{batch}-dr-wt-embwd-lr1e-3-wd0.3-post32-e16-v1"


def existing_named_experiment(name: str) -> str | None:
    payload = json.loads(
        command(
            [
                "beaker",
                "workspace",
                "experiments",
                WORKSPACE,
                "--text",
                name,
                "--format",
                "json",
            ]
        )
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [item for item in experiments if item.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple Beaker experiments already use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def spec_for(record: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    base = json.loads(
        command(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"])
    )
    if len(base.get("tasks", [])) != 1:
        raise RuntimeError("trusted Dense-1B source must contain exactly one task")
    spec = copy.deepcopy(base)
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted source is missing the Weka mount")
    task["name"] = "main"
    task["arguments"] = ["python", RUNNER, "--manifest", str(record["manifest"])]
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (
            str(variable.get("name", "")).startswith("BEAKER_")
            and variable.get("name") != "BEAKER_TOKEN"
        )
    ]
    set_env(task, "GIT_REF", revision)
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": True}
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Dense-1B sealed Pool-3B BS{record['batchSequences']} DR+WT+EmbedWD; "
        "LR1e-3 WD0.3; PD checkpoints every E4 through E32; POST E32,E48,E64,..."
    )
    return spec


def create(record: dict[str, Any], revision: str, priority: str) -> str:
    name = experiment_name(int(record["batchSequences"]))
    existing = existing_named_experiment(name)
    if existing:
        return existing
    spec = spec_for(record, revision, priority)
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("Beaker submission succeeded without a parsed experiment ID")
    return identifiers[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    validate_revision(args.revision)
    report = json.loads(REPORT.read_text())
    runs = [run for run in report.get("runs", []) if run.get("policy") == POLICY]
    if len(runs) != 2 or {int(run["batchSequences"]) for run in runs} != {64, 128}:
        raise RuntimeError("report must contain exactly the authorized BS64/BS128 coordinates")
    for run in runs:
        if run.get("experiment"):
            continue
        if args.print_only:
            print(json.dumps(spec_for(run, args.revision, args.priority), indent=2))
            continue
        if not args.submit_if_ready:
            print(f"BS{run['batchSequences']}: ready; pass --submit-if-ready to launch")
            continue
        experiment = create(run, args.revision, args.priority)
        run.update(
            {
                "status": "submitted",
                "beakerStatus": "submitted",
                "experiment": experiment,
                "beaker": experiment,
                "revision": args.revision,
                "activeEpoch": 1,
                "activePhase": "pending",
            }
        )
        print(f"BS{run['batchSequences']}: {experiment}")
    write_report(report)


if __name__ == "__main__":
    main()
