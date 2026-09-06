#!/usr/bin/env python3
"""Submit and register isolated Dense-1B DR+WT+EmbedWD re-decays."""

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
POLICY = "dense_1b_dr_wt_embwd_redecay_retry01_v1"
RUNNER = Path("scripts/models/run_dense_1b_dr_wt_embwd_redecay.py")
MANIFEST = Path("scripts/models/manifests/dense-1b-dr-wt-embwd-redecay-retry01.json")
REPORT = Path("reports/0802/data/wsd_data_loader_1b.json")
BASE_EXPERIMENTS = {
    128: "01M0WWNWS66NRG21QSKB87BKE7",
    256: "01M0WWNZMA0JC1P332QGNXWDKS",
}


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments, check=True, input=input_text, capture_output=True, text=True
    ).stdout


def load_run(run_id: str) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    matches = [item for item in manifest["runs"] if item["id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown re-decay run {run_id}")
    subprocess.run(
        [sys.executable, str(RUNNER), "--manifest", str(MANIFEST), "--run-id", run_id, "--validate-only"],
        check=True,
    )
    return matches[0]


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full commit hash")
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], check=True)
    upstream = command(["git", "rev-parse", "@{upstream}"]).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", revision, upstream], check=False).returncode:
        raise RuntimeError(f"revision {revision} is not reachable from pushed upstream {upstream}")


def set_revision(task: dict[str, Any], revision: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            return
    raise RuntimeError("trusted source task has no GIT_REF")


def experiment_name(item: dict[str, Any]) -> str:
    return "dense-1b-dr-wt-embwd-" + str(item["id"])


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    base = json.loads(command(["beaker", "experiment", "spec", BASE_EXPERIMENTS[int(item["batchSequences"])], "--format", "json"]))
    spec = copy.deepcopy(base)
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted source experiment must have exactly one task")
    task = spec["tasks"][0]
    task["arguments"] = [
        "python",
        str(RUNNER),
        "--manifest",
        str(MANIFEST),
        "--run-id",
        str(item["id"]),
    ]
    blocked = {"GANTRY_USE_TORCHRUN", "GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        variable for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (str(variable.get("name", "")).startswith("BEAKER_") and variable.get("name") != "BEAKER_TOKEN")
    ]
    set_revision(task, revision)
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "hostNetworking", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Isolated retry01 10% WSD re-decay and matched heldout evaluation for Dense-1B "
        f"BS{item['batchSequences']} DR+WT+EmbedWD LR{item['learningRate']} WD{item['weightDecay']} "
        f"from exact E{item['epoch']} PD. One node/eight GPUs; minRuntime omitted."
    )
    if "minRuntime" in task["context"]:
        raise AssertionError("re-decay task context must omit minRuntime")
    return spec


def audit_name(name: str) -> None:
    payload = json.loads(command(["beaker", "workspace", "experiments", WORKSPACE, "--text", name, "--format", "json"]))
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    if any(item.get("name") == name for item in experiments):
        raise RuntimeError(f"refusing duplicate Beaker experiment name {name}")


def write_report(item: dict[str, Any], experiment: str, revision: str) -> None:
    report = json.loads(REPORT.read_text())
    retries = report.setdefault("redecayRetries", [])
    if any(record.get("id") == item["id"] for record in retries):
        raise RuntimeError(f"re-decay {item['id']} is already registered")
    retries.append(
        {
            "id": item["id"],
            "policy": POLICY,
            "status": "submitted",
            "batchSequences": item["batchSequences"],
            "lr": item["learningRate"],
            "wd": item["weightDecay"],
            "epoch": item["epoch"],
            "sourceCheckpoint": item["sourceCheckpoint"],
            "output": item["retryOutput"],
            "endpointCheckpoint": item["retryOutput"] + f"/step{item['endpointStep']}-retry01",
            "expectedRuntimeSeconds": item["expectedRuntimeSeconds"],
            "decayFraction": 0.1,
            "retry": "retry01",
            "originalValidationExact": item["originalValidationExact"],
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "gpuCount": 8,
            "nodeCount": 1,
            "automaticTaskRetries": 8,
            "minRuntimeOmitted": True,
            "reason": "Submitted isolated retry01 re-decay from the exact retained PD checkpoint; original POST output is immutable.",
        }
    )
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.with_suffix(".js").write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.print_only and args.register:
        raise SystemExit("--print-only cannot be combined with --register")
    item = load_run(args.run_id)
    validate_revision(args.revision)
    name = experiment_name(item)
    audit_name(name)
    spec = build_spec(item, args.revision, args.priority)
    if args.print_only:
        print(json.dumps(spec, indent=2))
        return
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    print(output, end="")
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission succeeded without a parsed experiment ID")
    if args.register:
        write_report(item, identifiers[0], args.revision)


if __name__ == "__main__":
    main()
