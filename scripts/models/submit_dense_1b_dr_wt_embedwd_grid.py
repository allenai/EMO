#!/usr/bin/env python3
"""Register and conditionally submit the Dense-1B BS128/256 DR+WT+EmbedWD grids."""

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
REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
SMALL_REPORTS = (
    Path("reports/0802/data/wsd_batch_size_474m.json"),
    Path("reports/0802/data/wsd_batch_size_153m.json"),
)
MODEL_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
THRESHOLD = 5
WD_TUNING_POLICY_HOLD = "held_by_locked_wd_predecay_policy_2026_08_21"
CONFIGS: dict[int, dict[str, Any]] = {
    128: {
        "baseExperiment": "01M02YR7SFKQ2C9ABV15QK7VF5",
        "manifest": "scripts/models/manifests/dense-1b-bs128-dr-wt-embwd-grid.json",
        "coordinates": (("1e-3", "0.3"), ("1e-3", "1.0")),
    },
    256: {
        "baseExperiment": "01KZD997FY92SE7CQM504CVDJW",
        "manifest": "scripts/models/manifests/dense-1b-bs256-dr-wt-embwd-grid.json",
        "coordinates": (
            ("1e-3", "0.333"),
            ("1e-3", "1.0"),
            ("2e-3", "0.333"),
            ("2e-3", "1.0"),
        ),
    },
}


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        input=input_text,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def output_for(batch: int, lr: str, wd: str) -> str:
    return f"{MODEL_ROOT}/bs{batch}_dr_wt_embwd_lr{lr}_wd{wd}"


def column_key(batch: int) -> str:
    return f"drwtembwd{batch}"


def chain_id(batch: int) -> str:
    return f"dense-1b-bs{batch}-dr-wt-embwd-grid"


def run_id(batch: int, lr: str, wd: str) -> str:
    return f"drwtembwd{batch}-lr{lr}-wd{wd}"


def insert_column(report: dict[str, Any], batch: int) -> None:
    key = column_key(batch)
    if any(column.get("key") == key for column in report.get("columns", [])):
        return
    column = {
        "key": key,
        "label": f"BS{batch} · DR+WT+EmbedWD",
        "batchSequences": batch,
        "color": "#ea580c" if batch == 128 else "#c2410c",
        "dynamicRepacking": True,
        "weightTying": True,
        "decayEmbeddings": True,
    }
    columns = report.setdefault("columns", [])
    predecessor = f"dr{batch}"
    index = next(
        (
            position
            for position, candidate in enumerate(columns)
            if candidate.get("key") == predecessor
        ),
        len(columns) - 1,
    )
    columns.insert(index + 1, column)


def register_plan(report: dict[str, Any]) -> None:
    chains = report.setdefault("drWtEmbedWdGridChains", [])
    runs = report.setdefault("runs", [])
    existing_chains = {record.get("id"): record for record in chains}
    existing_runs = {record.get("id"): record for record in runs}
    for batch, config in CONFIGS.items():
        insert_column(report, batch)
        identifier = chain_id(batch)
        if identifier not in existing_chains:
            record = {
                "id": identifier,
                "label": f"BS{batch} · DR+WT+EmbedWD LR/WD grid",
                "variant": "DR+WT+EmbedWD",
                "batchSequences": batch,
                "globalBatchTokens": batch * 4096,
                "rankMicrobatchSequences": 8,
                "gpuCount": 8,
                "gradientAccumulation": batch // 64,
                "coordinates": [{"lr": lr, "wd": wd} for lr, wd in config["coordinates"]],
                "initialTargets": [1, 2, 4],
                "epochIncrement": 4,
                "maxEpoch": 64,
                "dataOrder": "dynamic_repacking_from_e2",
                "dynamicRepacking": True,
                "weightTying": True,
                "decayEmbeddings": True,
                "embeddingWeightDecay": "global",
                "status": "planned",
                "triggerThreshold": THRESHOLD,
                "completedSmallChainsAtLastCheck": 0,
                "manifest": config["manifest"],
                "baseExperiment": config["baseExperiment"],
                "automaticTaskRetries": 8,
                "stopOnNonImprovement": True,
                "results": {},
                "frontiers": {},
                "reason": (
                    "Deferred until at least five of the ten small-model logical chains "
                    "finish successfully. One persistent Beaker job evaluates every "
                    "requested LR/WD coordinate while preserving separate canonical "
                    "optimizer trajectories."
                ),
            }
            chains.append(record)
            existing_chains[identifier] = record
        for lr, wd in config["coordinates"]:
            coordinate_id = run_id(batch, lr, wd)
            if coordinate_id in existing_runs:
                continue
            coordinate = {
                "id": coordinate_id,
                "method": column_key(batch),
                "label": f"BS{batch} DR+WT+EmbedWD LR{lr} WD{wd}",
                "batchSequences": batch,
                "globalBatchTokens": batch * 4096,
                "lr": lr,
                "wd": wd,
                "weightTying": True,
                "decayEmbeddings": True,
                "dynamicRepacking": True,
                "dataOrder": "dynamic_repacking_from_e2",
                "rankMicrobatchSequences": 8,
                "gpuCount": 8,
                "nodeCount": 1,
                "gradientAccumulation": batch // 64,
                "status": "planned",
                "activeEpoch": 1,
                "sourceExperiment": config["baseExperiment"],
                "sourceCheckpoint": "fresh",
                "output": output_for(batch, lr, wd),
                "results": {},
                "reason": f"Planned coordinate in the deferred BS{batch} persistent grid.",
            }
            runs.append(coordinate)
            existing_runs[coordinate_id] = coordinate
    setup_note = (
        " BS128 and BS256 DR+WT+EmbedWD grids use rank microbatch 8 on one 8-GPU "
        "node; gradient accumulation is 2 and 4 respectively. Every LR/WD trajectory "
        "stays in its own canonical output directory across epochs."
    )
    if setup_note.strip() not in str(report.get("setup", "")):
        report["setup"] = str(report.get("setup", "")).rstrip() + setup_note
    selection_note = (
        " The BS128/256 DR+WT+EmbedWD columns compare every explicitly requested LR/WD "
        "tuple at each frontier and stop after the best held-out CE fails to improve."
    )
    if selection_note.strip() not in str(report.get("selection", "")):
        report["selection"] = str(report.get("selection", "")).rstrip() + selection_note


def successful_small_chains() -> tuple[int, list[str]]:
    completed: list[str] = []
    for path in SMALL_REPORTS:
        report = json.loads(path.read_text())
        model = "474m" if "474m" in path.name else "153m"
        original = [
            record
            for record in report.get("batchSweeps", [])
            if int(record.get("batchSequences", 0)) == 32 and record.get("experiment")
        ]
        if original and str(original[-1].get("status", "")).lower() == "complete":
            completed.append(f"{model}-bs32")
        for record in report.get("adaptiveDrWtEmbedWdChains", []):
            if str(record.get("status", "")).lower() == "complete":
                completed.append(f"{model}-bs{record['batchSequences']}-drwtembwd")
    return len(completed), completed


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("--revision must be a full 40-character commit hash")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"],
        check=True,
    )


def set_revision(task: dict[str, Any], revision: str) -> None:
    for variable in task.get("envVars", []):
        if variable.get("name") == "GIT_REF":
            variable["value"] = revision
            return
    raise RuntimeError("trusted source task has no GIT_REF")


def build_spec(batch: int, revision: str, priority: str) -> dict[str, Any]:
    config = CONFIGS[batch]
    base = json.loads(
        run(["beaker", "experiment", "spec", config["baseExperiment"], "--format", "json"])
    )
    tasks = base.get("tasks", [])
    if len(tasks) != 1:
        raise RuntimeError("trusted 1B source must contain exactly one task")
    spec = copy.deepcopy(base)
    task = spec["tasks"][0]
    mount_paths = {dataset.get("mountPath") for dataset in task.get("datasets", [])}
    if "/weka/oe-training-default" not in mount_paths:
        raise RuntimeError("trusted source is missing the Weka mount")
    if not task.get("image"):
        raise RuntimeError("trusted source has no image")
    task["arguments"] = [
        "python",
        "scripts/models/run_dense_1b_dr_wt_embedwd_grid.py",
        "--manifest",
        config["manifest"],
    ]
    blocked = {
        "GANTRY_USE_TORCHRUN",
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "NUM_NODES",
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
    set_revision(task, revision)
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in (
        "replicas",
        "leaderSelection",
        "hostNetworking",
        "synchronizedStartTimeout",
    ):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Dense-1B BS{batch} DR+WT+EmbedWD LR/WD grid. E1 uses ordinary packing, "
        "DR begins at E2, each coordinate keeps one canonical output, and the chain "
        "stops on held-out validation saturation."
    )
    return spec


def experiment_name(batch: int) -> str:
    return f"dense-1b-bs{batch}-dr-wt-embwd-lr-wd-grid"


def existing_named_experiment(name: str) -> str | None:
    payload = json.loads(
        run(["beaker", "workspace", "experiments", WORKSPACE, "--text", name, "--format", "json"])
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [item for item in experiments if item.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple Beaker experiments already use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def create_experiment(batch: int, revision: str, priority: str) -> str:
    name = experiment_name(batch)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    spec = build_spec(batch, revision, priority)
    output = run(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("Beaker submission succeeded without a parsed experiment ID")
    return identifiers[0]


def register_submission(
    report: dict[str, Any], batch: int, experiment: str, revision: str, completed_count: int
) -> None:
    chain = next(
        record for record in report["drWtEmbedWdGridChains"] if record["id"] == chain_id(batch)
    )
    if chain.get("experiment") not in {None, experiment}:
        raise RuntimeError(f"BS{batch} grid already points to another experiment")
    chain.update(
        {
            "status": "submitted",
            "beakerStatus": "submitted",
            "activeEpoch": 1,
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "completedSmallChainsAtSubmission": completed_count,
            "reason": (
                f"Submitted after {completed_count} of ten small-model chains finished. "
                "All requested coordinates run inside one persistent retry-safe job."
            ),
        }
    )
    for lr, wd in CONFIGS[batch]["coordinates"]:
        coordinate = next(
            record for record in report["runs"] if record["id"] == run_id(batch, lr, wd)
        )
        coordinate.update(
            {
                "status": "submitted",
                "activeEpoch": 1,
                "experiment": experiment,
                "beaker": experiment,
                "revision": revision,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--priority", default="urgent")
    args = parser.parse_args()
    if args.register_only == args.submit_if_ready:
        parser.error("choose exactly one of --register-only or --submit-if-ready")
    report = json.loads(REPORT_PATH.read_text())
    register_plan(report)
    completed_count, _ = successful_small_chains()
    for chain in report["drWtEmbedWdGridChains"]:
        chain["completedSmallChainsAtLastCheck"] = completed_count
    write_report(report)
    if args.register_only:
        print(f"registered deferred 1B grids; completed_small_chains={completed_count}/10")
        return
    # The 2026-08-21 policy stops launching new WD sweeps. These grids have no
    # previously selected WD to lock, so they stay registered for provenance
    # but may not be submitted by the old completion-count trigger.
    for chain in report["drWtEmbedWdGridChains"]:
        if not chain.get("experiment"):
            chain.update(
                {
                    "status": "held",
                    "policyHold": WD_TUNING_POLICY_HOLD,
                    "reason": (
                        "Submission is held because the active policy forbids new WD "
                        "tuning and Dense-1B has no previously selected WD to lock."
                    ),
                }
            )
    for coordinate in report.get("runs", []):
        if coordinate.get("method") in {"drwtembwd128", "drwtembwd256"} and not coordinate.get(
            "experiment"
        ):
            coordinate["status"] = "held"
            coordinate["policyHold"] = WD_TUNING_POLICY_HOLD
    write_report(report)
    print(
        "1B grid trigger held by locked-WD pre-decay policy: no Dense-1B WD was "
        "previously selected, so no LR/WD grid will be submitted"
    )


if __name__ == "__main__":
    main()
