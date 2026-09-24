#!/usr/bin/env python3
"""Prepare or submit eight Table-5 replications on independent pools B/C.

Submission is opt-in via ``--submit``.  The default mode performs local plan
validation and prints the exact coordinates and output ownership map without
contacting Beaker.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import run_dense_1b_independent_pool_table5 as runner
import submit_dense_1b_dr_wt_embedwd_grid as beaker_base


WORKSPACE = "ai2/flex2"
BASE_EXPERIMENT = "01KZD997FY92SE7CQM504CVDJW"
RUNNER = "scripts/models/run_dense_1b_independent_pool_table5.py"
PLAN = Path("scripts/models/manifests/dense-1b-independent-pools-table5-v1.json")


def name_for(item: dict[str, Any]) -> str:
    return (
        f"dense-1b-pool-{item['pool']}-bs{item['batchSequences']}-original-"
        f"lr{item['lr']}-wd{item['wd']}-table5-replication-v1"
    )


def build_spec(
    item: dict[str, Any], *, revision: str, priority: str, base_experiment: str
) -> dict[str, Any]:
    spec = beaker_base.spec_for(
        base_experiment=base_experiment,
        manifest=str(PLAN),
        lr=str(item["lr"]),
        wd=str(item["wd"]),
        revision=revision,
        priority=priority,
        finalize_only=False,
        description=(
            f"Table-5 1.5B Original replication on independent train-only pool "
            f"{str(item['pool']).upper()}: BS{item['batchSequences']}, LR{item['lr']}, "
            f"WD{item['wd']}, terminal E{item['targetEpoch']}. Ordinary shuffled data, "
            "untied embeddings, zero embedding WD, and one retained checkpoint per epoch."
        ),
    )
    task = spec["tasks"][0]
    task["arguments"] = [
        "python",
        RUNNER,
        "--plan",
        str(PLAN),
        "--pool",
        str(item["pool"]),
        "--batch",
        str(item["batchSequences"]),
    ]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    # Absence of minRuntime is intentional: all eight jobs are unallocated.
    task["context"] = {"priority": priority, "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "hostNetworking", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    return spec


def submit(spec: dict[str, Any], name: str) -> str:
    completed = subprocess.run(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        check=True,
        input=json.dumps(spec),
        text=True,
        capture_output=True,
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", completed.stdout)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--base-experiment", default=BASE_EXPERIMENT)
    parser.add_argument("--revision")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    runner.validate_plan(plan)
    for manifest in (plan["data"]["poolB"], plan["data"]["poolC"]):
        if not Path(manifest).is_file():
            raise FileNotFoundError(f"missing materialized train-pool manifest {manifest}")

    if args.print_specs or args.submit:
        if not args.revision:
            parser.error("--revision is required with --print-specs or --submit")
        beaker_base.validate_revision(args.revision)

    for item in plan["trajectories"]:
        name = name_for(item)
        if args.print_specs or args.submit:
            spec = build_spec(
                item,
                revision=args.revision,
                priority=args.priority,
                base_experiment=args.base_experiment,
            )
            if args.print_specs:
                print(json.dumps({"name": name, "spec": spec}, indent=2))
            else:
                print(f"{name}: {submit(spec, name)}")
        else:
            print(
                f"pool={item['pool']} bs={item['batchSequences']} lr={item['lr']} "
                f"wd={item['wd']} terminal=E{item['targetEpoch']} output={item['output']}"
            )


if __name__ == "__main__":
    main()
