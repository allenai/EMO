#!/usr/bin/env python3
"""Submit the two fresh exact-WD 474M Pool-3B BS256 E16/E32 probes."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_474m_pool3b_bs256_wd_probe as runner

WORKSPACE = "ai2/flex2"
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
RUNNER = "scripts/models/run_dense_474m_pool3b_bs256_wd_probe.py"


def command(arguments: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(arguments, input=input_text, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def clean_spec(spec: dict[str, Any], item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("trusted base experiment must contain exactly one task")
    task = spec["tasks"][0]
    blocked = {
        "GANTRY_USE_TORCHRUN",
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "NUM_NODES",
        "GIT_REF",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in blocked
        and not (
            str(variable.get("name", "")).startswith("BEAKER_")
            and variable.get("name") != "BEAKER_TOKEN"
        )
    ]
    task["envVars"].extend(
        [
            {"name": "GIT_REF", "value": revision},
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        ]
    )
    task["arguments"] = [
        "python",
        RUNNER,
        "--manifest",
        str(runner.DEFAULT_MANIFEST),
        "--coordinate",
        str(item["id"]),
    ]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "autoResume": True}
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Fresh exact 474M Pool-3B BS256 LR2e-3 WD{item['weightDecay']} trajectory; "
        "produce its own Pool-1B E1 source, bridge to the sealed disjoint Pool-3B lineage, "
        "then retain PD and run isolated 10% WSD plus heldout evaluation at E16 and E32; "
        "one eight-GPU node, rank microbatch 16, gradient accumulation 2, expandable CUDA "
        "segments, auto-resume, eight retries, and one writer per canonical output."
    )
    return spec


def create(name: str, spec: dict[str, Any]) -> str:
    output = command(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID: {output}")
    return identifiers[0]


def write_report(records: list[dict[str, Any]], revision: str) -> None:
    report = json.loads(REPORT.read_text())
    if report.get("weightDecayProbes474m"):
        raise RuntimeError("474M WD probes are already registered")
    now = datetime.now(tz=UTC).isoformat()
    report["weightDecayProbes474m"] = {
        "policy": runner.POLICY,
        "status": "submitted",
        "revision": revision,
        "submittedAt": now,
        "evaluationEpochs": [16, 32],
        "coordinates": records,
    }
    report["updatedAt"] = now
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    if args.print_specs == args.submit_if_ready:
        raise SystemExit("select exactly one of --print-specs or --submit-if-ready")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full commit hash")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.revision, "origin/sewonm/icsl"],
        check=True,
    )
    config = json.loads(runner.DEFAULT_MANIFEST.read_text())
    records = []
    for item in config["coordinates"]:
        base = json.loads(
            command(["beaker", "experiment", "spec", item["baseExperiment"], "--format", "json"])
        )
        spec = clean_spec(copy.deepcopy(base), item, args.revision, args.priority)
        if args.print_specs:
            print(json.dumps(spec, indent=2))
            continue
        name = f"{item['id']}-e16-e32"
        experiment = create(name, spec)
        records.append(
            {
                "id": item["id"],
                "model": "474m",
                "pool": "dclm3b",
                "batchSequences": 256,
                "learningRate": "2e-3",
                "weightDecay": item["weightDecay"],
                "sourceOutput": item["sourceOutput"],
                "output": item["output"],
                "status": "submitted",
                "experiment": experiment,
                "revision": args.revision,
                "gpuCount": 8,
                "rankMicrobatchSequences": 16,
                "gradientAccumulationSteps": 2,
                "minRuntimeOmitted": True,
                "resolvedCheckpointEpochs": [],
                "resolvedPostEpochs": [],
            }
        )
        print(f"{item['id']}: {experiment}")
    if args.submit_if_ready:
        write_report(records, args.revision)


if __name__ == "__main__":
    main()
