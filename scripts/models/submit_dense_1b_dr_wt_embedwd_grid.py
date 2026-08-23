#!/usr/bin/env python3
"""Register and guard Dense-1B all-POST coordinate experiments.

The six primary BS128/256 coordinates are released only after all ten small
logical chains are terminal with confirmed POST saturation.  Each coordinate
is a separate one-node/eight-GPU Beaker experiment.  A conditional three-job
BS512 follow-up is released only when the terminal BS256 winner uses LR 2e-3.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

WORKSPACE = "ai2/flex2"
POLICY = "dense_1b_all_postdecay_saturation_v1"
SMALL_POLICIES = {
    "locked_wd_predecay_saturation_v1",
    "locked_wd_requested_postdecay_finalizer_v1",
}
REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
SMALL_REPORTS = (
    Path("reports/0802/data/wsd_batch_size_474m.json"),
    Path("reports/0802/data/wsd_batch_size_153m.json"),
)
MODEL_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
TRIGGER_THRESHOLD = 10
PRIMARY: dict[int, dict[str, Any]] = {
    128: {
        "baseExperiment": "01M02YR7SFKQ2C9ABV15QK7VF5",
        "manifest": "scripts/models/manifests/dense-1b-bs128-dr-wt-embwd-grid.json",
        "variant": "DR+WT+EmbedWD",
        "coordinates": (("1e-3", "0.3"), ("1e-3", "1.0")),
    },
    256: {
        "baseExperiment": "01KZD997FY92SE7CQM504CVDJW",
        "manifest": "scripts/models/manifests/dense-1b-bs256-dr-wt-embwd-grid.json",
        "variant": "DR+WT+EmbedWD",
        "coordinates": (
            ("1e-3", "0.3"),
            ("1e-3", "1.0"),
            ("2e-3", "0.3"),
            ("2e-3", "1.0"),
        ),
    },
}
CONDITIONAL: tuple[dict[str, Any], ...] = (
    {
        "runId": "conditional-original512-lr2e-3-wd0.333",
        "variant": "Original",
        "batch": 512,
        "lr": "2e-3",
        "wd": "0.333",
        "baseExperiment": "01KZGV0B3Q7ESNHR83KVQJRRE8",
        "manifest": "scripts/models/manifests/dense-1b-bs512-original-lr2e-3-wd0.333-pdpost.json",
        "sourceCheckpoint": (
            "/weka/oe-training-default/sewonm/icsl/models/"
            "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_e8_lr2e-3_wd0.333_"
            "warmup48_e8_lr2_wd0333_r30/step3432"
        ),
    },
    {
        "runId": "conditional-drwtembwd512-lr2e-3-wd0.333",
        "variant": "DR+WT+EmbedWD",
        "batch": 512,
        "lr": "2e-3",
        "wd": "0.333",
        "baseExperiment": "01KZD997FY92SE7CQM504CVDJW",
        "manifest": "scripts/models/manifests/dense-1b-bs512-dr-wt-embwd-lr2e-3-grid.json",
        "sourceCheckpoint": "fresh",
    },
    {
        "runId": "conditional-drwtembwd512-lr2e-3-wd1.0",
        "variant": "DR+WT+EmbedWD",
        "batch": 512,
        "lr": "2e-3",
        "wd": "1.0",
        "baseExperiment": "01KZD997FY92SE7CQM504CVDJW",
        "manifest": "scripts/models/manifests/dense-1b-bs512-dr-wt-embwd-lr2e-3-grid.json",
        "sourceCheckpoint": "fresh",
    },
)


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        arguments, check=True, input=input_text, capture_output=True, text=True
    )
    return completed.stdout


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def column_key(batch: int) -> str:
    return f"drwtembwd{batch}"


def chain_id(batch: int) -> str:
    return f"dense-1b-bs{batch}-dr-wt-embwd-grid"


def run_id(batch: int, lr: str, wd: str) -> str:
    return f"drwtembwd{batch}-lr{lr}-wd{wd}"


def output_for(batch: int, lr: str, wd: str, variant: str = "DR+WT+EmbedWD") -> str:
    if batch == 512 and variant == "Original":
        return f"{MODEL_ROOT}/bs512_original_lr2e-3_wd0.333_constant_lr_from_e8"
    return f"{MODEL_ROOT}/bs{batch}_dr_wt_embwd_lr{lr}_wd{wd}"


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
        (position for position, item in enumerate(columns) if item.get("key") == predecessor),
        len(columns) - 1,
    )
    columns.insert(index + 1, column)


def upsert(records: list[dict[str, Any]], identifier: str, value: dict[str, Any]) -> dict[str, Any]:
    matches = [record for record in records if record.get("id") == identifier]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate report record {identifier}")
    if matches:
        record = matches[0]
        preserved = {
            key: record[key]
            for key in (
                "experiment",
                "beaker",
                "revision",
                "status",
                "beakerStatus",
                "results",
                "preDecayResults",
                "postDecayResults",
                "postDecaySelection",
                "latestPostDecayContinuation",
                "attemptHistory",
                "wandbHealth",
                "activeEpoch",
                "activePhase",
                "activeSource",
                "activeOutput",
                "progress",
                "job",
                "jobs",
                "pruneRequested",
                "pruneDeferredReason",
                "higherWdWinEvidence",
                "stopIssued",
                "needsAttention",
                "selectedPostDecayEpoch",
                "selectedPostDecayValidationExact",
                "experimentsByCoordinate",
                "coordinateStates",
                "selectedRun",
                "selectedLr",
                "selectedWd",
                "selectedVariant",
                "completedSmallChainsAtSubmission",
            )
            if key in record
        }
        record.update(value)
        record.update(preserved)
        return record
    records.append(value)
    return value


def primary_coordinate_record(
    batch: int, lr: str, wd: str, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": run_id(batch, lr, wd),
        "method": column_key(batch),
        "label": f"BS{batch} DR+WT+EmbedWD LR{lr} WD{wd}",
        "variant": "DR+WT+EmbedWD",
        "policy": POLICY,
        "batchSequences": batch,
        "globalBatchTokens": batch * 4096,
        "lr": lr,
        "wd": wd,
        "weightTying": True,
        "decayEmbeddings": True,
        "dynamicRepacking": True,
        "dataOrder": "ordinary_e1_dynamic_repacking_from_e2",
        "rankMicrobatchSequences": 8,
        "gpuCount": 8,
        "nodeCount": 1,
        "gradientAccumulation": batch // 64,
        "status": "planned",
        "activeEpoch": 1,
        "sourceExperiment": config["baseExperiment"],
        "sourceCheckpoint": "fresh",
        "manifest": config["manifest"],
        "output": output_for(batch, lr, wd),
        "results": {},
        "preDecayResults": {},
        "postDecayResults": {},
        "comparisonPolicy": "post_decay_only",
        "postDecayStartEpoch": 8,
        "checkpointOnlyEpochs": [1, 2, 4],
        "postDecayEvaluation": "every_scheduled_frontier_from_e8",
        "postDecaySourceCount": 3,
        "postDecaySaturationCriterion": "strict_non_improvement",
        "reason": "Deferred coordinate; one persistent one-node/eight-GPU experiment.",
    }


def register_plan(report: dict[str, Any]) -> None:
    chains = report.setdefault("drWtEmbedWdGridChains", [])
    runs = report.setdefault("runs", [])
    guarded_ids = {
        run_id(batch, lr, wd)
        for batch, config in PRIMARY.items()
        for lr, wd in config["coordinates"]
    } | {item["runId"] for item in CONDITIONAL}
    launched_old_policy = [
        record["id"]
        for record in runs
        if record.get("id") in guarded_ids
        and record.get("experiment")
        and record.get("policy") != POLICY
    ]
    if launched_old_policy:
        raise RuntimeError(
            "cannot rewrite launched Dense-1B coordinates to the all-POST policy: "
            + ", ".join(launched_old_policy)
        )
    stale_ids = {
        "drwtembwd256-lr1e-3-wd0.333",
        "drwtembwd256-lr2e-3-wd0.333",
    }
    stale = [record for record in runs if record.get("id") in stale_ids]
    if any(record.get("experiment") for record in stale):
        raise RuntimeError("cannot replace launched BS256 WD0.333 coordinates with requested WD0.3")
    runs[:] = [record for record in runs if record.get("id") not in stale_ids]
    for batch, config in PRIMARY.items():
        insert_column(report, batch)
        chain = upsert(
            chains,
            chain_id(batch),
            {
                "id": chain_id(batch),
                "label": f"BS{batch} · DR+WT+EmbedWD independent all-POST coordinates",
                "variant": "DR+WT+EmbedWD",
                "policy": POLICY,
                "batchSequences": batch,
                "globalBatchTokens": batch * 4096,
                "rankMicrobatchSequences": 8,
                "gpuCountPerCoordinate": 8,
                "nodeCountPerCoordinate": 1,
                "gradientAccumulation": batch // 64,
                "coordinates": [{"lr": lr, "wd": wd} for lr, wd in config["coordinates"]],
                "initialTargets": [1, 2, 4],
                "epochIncrement": 4,
                "maxEpoch": 256,
                "dataOrder": "ordinary_e1_dynamic_repacking_from_e2",
                "dynamicRepacking": True,
                "weightTying": True,
                "decayEmbeddings": True,
                "status": "planned",
                "triggerThreshold": TRIGGER_THRESHOLD,
                "completedSmallChainsAtLastCheck": 0,
                "manifest": config["manifest"],
                "automaticTaskRetries": 8,
                "parallelCoordinates": len(config["coordinates"]),
                "comparisonPolicy": "post_decay_only",
                "postDecayStartEpoch": 8,
                "checkpointOnlyEpochs": [1, 2, 4],
                "postDecayEvaluation": "every_scheduled_frontier_from_e8",
                "postDecaySourceCount": 3,
                "postDecaySaturationCriterion": "strict_non_improvement",
                "crossWdPruning": (
                    "stop_lower_wd_after_higher_wd_strictly_wins_same_lr_post_epoch"
                ),
                "experimentsByCoordinate": {},
                "reason": (
                    "Held until all ten small-model chains have confirmed POST saturation; "
                    "then every coordinate launches as its own one-node/eight-GPU job."
                ),
            },
        )
        chain.pop("policyHold", None)
        chain.pop("experiment", None)
        chain.pop("beaker", None)
        chain.pop("stopOnNonImprovement", None)
        for lr, wd in config["coordinates"]:
            record = upsert(
                runs,
                run_id(batch, lr, wd),
                primary_coordinate_record(batch, lr, wd, config),
            )
            record.pop("policyHold", None)
    conditional_chain = upsert(
        chains,
        "dense-1b-bs512-lr2e-3-conditional-followup",
        {
            "id": "dense-1b-bs512-lr2e-3-conditional-followup",
            "label": "Conditional BS512 LR2e-3 follow-up",
            "variant": "Original + DR+WT+EmbedWD",
            "policy": POLICY,
            "batchSequences": 512,
            "status": "conditional_held",
            "trigger": "terminal_bs256_selected_lr_2e-3",
            "coordinates": [
                {"variant": item["variant"], "lr": item["lr"], "wd": item["wd"]}
                for item in CONDITIONAL
            ],
            "gpuCountPerCoordinate": 8,
            "nodeCountPerCoordinate": 1,
            "gradientAccumulation": 8,
            "experimentsByCoordinate": {},
            "comparisonPolicy": "post_decay_only",
            "postDecayStartEpoch": 8,
            "checkpointOnlyEpochs": [1, 2, 4],
            "postDecayEvaluation": "every_scheduled_frontier_from_e8",
            "postDecaySourceCount": 3,
            "postDecaySaturationCriterion": "strict_non_improvement",
            "reason": "Held unless the terminal BS256 winner uses LR2e-3.",
        },
    )
    for item in CONDITIONAL:
        upsert(
            runs,
            item["runId"],
            {
                "id": item["runId"],
                "method": "conditional512",
                "label": f"Conditional BS512 {item['variant']} LR{item['lr']} WD{item['wd']}",
                "variant": item["variant"],
                "policy": POLICY,
                "batchSequences": 512,
                "globalBatchTokens": 512 * 4096,
                "lr": item["lr"],
                "wd": item["wd"],
                "weightTying": item["variant"] == "DR+WT+EmbedWD",
                "decayEmbeddings": item["variant"] == "DR+WT+EmbedWD",
                "dynamicRepacking": item["variant"] == "DR+WT+EmbedWD",
                "dataOrder": (
                    "ordinary_e1_dynamic_repacking_from_e2"
                    if item["variant"] == "DR+WT+EmbedWD"
                    else "ordinary_shuffled"
                ),
                "rankMicrobatchSequences": 8,
                "gpuCount": 8,
                "nodeCount": 1,
                "gradientAccumulation": 8,
                "status": "conditional_held",
                "manifest": item["manifest"],
                "sourceExperiment": item["baseExperiment"],
                "sourceCheckpoint": item["sourceCheckpoint"],
                "output": output_for(512, item["lr"], item["wd"], item["variant"]),
                "results": {},
                "preDecayResults": {},
                "postDecayResults": {},
                "comparisonPolicy": "post_decay_only",
                "postDecayStartEpoch": 8,
                "checkpointOnlyEpochs": [1, 2, 4],
                "postDecayEvaluation": "every_scheduled_frontier_from_e8",
                "postDecaySourceCount": 3,
                "postDecaySaturationCriterion": "strict_non_improvement",
                "reason": "Held unless the terminal BS256 winner uses LR2e-3.",
            },
        )
    conditional_chain.pop("policyHold", None)
    report["dense1bPdPostPolicy"] = {
        "policy": POLICY,
        "primaryLaunchCondition": "all_10_small_chains_confirmed_post_saturated",
        "parallelPrimaryCoordinates": 6,
        "gpuCountPerCoordinate": 8,
        "checkpointOnlyEpochs": [1, 2, 4],
        "postDecayStartEpoch": 8,
        "preDecayRole": "checkpoint_and_report_provenance_only",
        "postDecayComparison": "ordered_all_scheduled_frontiers_from_e8",
        "saturationDecisionGroup": "post_decay_only",
        "crossWdPruning": (
            "Within a fixed BS/LR and the same completed POST epoch, WD1.0 strictly "
            "beating the lower WD stops lower-WD constant-LR training after at least "
            "three POST results; its latest three eligible checkpoints remain POST-decayed "
            "and evaluated before the coordinate is marked pruned."
        ),
        "conditionalBs512": "launch_only_if_terminal_bs256_selected_lr_is_2e-3",
    }
    old_selection_note = (
        " The BS128/256 DR+WT+EmbedWD columns compare every explicitly requested LR/WD "
        "tuple at each frontier and stop after the best held-out CE fails to improve."
    )
    new_selection_note = (
        " The guarded BS128/256 DR+WT+EmbedWD phase runs six independent one-node "
        "coordinates. E1/E2/E4 constant-LR checkpoints are retained without POST decay; "
        "every scheduled frontier from E8 is independently decayed and evaluated. "
        "Saturation and matched-WD pruning use only POST CE."
    )
    selection = str(report.get("selection", "")).replace(old_selection_note, "")
    if new_selection_note.strip() not in selection:
        selection = selection.rstrip() + new_selection_note
    report["selection"] = selection
    setup_note = (
        " Each guarded Dense-1B LR/WD coordinate is a separate one-node, eight-GPU "
        "persistent experiment so cross-WD pruning can stop only the losing coordinate."
    )
    setup = str(report.get("setup", ""))
    if setup_note.strip() not in setup:
        setup = setup.rstrip() + setup_note
    report["setup"] = setup


def successful_small_chains() -> tuple[int, list[str]]:
    completed: list[str] = []
    for path in SMALL_REPORTS:
        report = json.loads(path.read_text())
        model = "474m" if "474m" in path.name else "153m"
        originals = [
            record
            for record in report.get("batchSweeps", [])
            if int(record.get("batchSequences", 0)) == 32 and record.get("experiment")
        ]
        if len(originals) != 1:
            raise RuntimeError(f"{model} must have exactly one registered original BS32 chain")
        if str(originals[0].get("status", "")).lower() == "complete":
            completed.append(f"{model}-bs32")
        for record in report.get("adaptiveDrWtEmbedWdChains", []):
            if record.get("policy") not in SMALL_POLICIES:
                continue
            selection = record.get("postDecaySelection") or {}
            if (
                str(record.get("status", "")).lower() == "complete"
                and selection.get("status") == "complete"
                and selection.get("postDecaySaturated") is True
            ):
                completed.append(f"{model}-bs{record['batchSequences']}-locked-wd")
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


def spec_for(
    *,
    base_experiment: str,
    manifest: str,
    lr: str,
    wd: str,
    revision: str,
    priority: str,
    finalize_only: bool,
    description: str,
) -> dict[str, Any]:
    base = json.loads(run(["beaker", "experiment", "spec", base_experiment, "--format", "json"]))
    tasks = base.get("tasks", [])
    if len(tasks) != 1:
        raise RuntimeError("trusted Dense-1B source must contain exactly one task")
    spec = copy.deepcopy(base)
    task = spec["tasks"][0]
    if "/weka/oe-training-default" not in {
        dataset.get("mountPath") for dataset in task.get("datasets", [])
    }:
        raise RuntimeError("trusted source is missing the Weka mount")
    arguments = [
        "python",
        "scripts/models/run_dense_1b_dr_wt_embedwd_grid.py",
        "--manifest",
        manifest,
        "--lr",
        lr,
        "--wd",
        wd,
    ]
    if finalize_only:
        arguments.append("--finalize-only")
    task["arguments"] = arguments
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
    set_revision(task, revision)
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for key in ("replicas", "leaderSelection", "hostNetworking", "synchronizedStartTimeout"):
        task.pop(key, None)
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = description
    return spec


def existing_named_experiment(name: str) -> str | None:
    payload = json.loads(
        run(["beaker", "workspace", "experiments", WORKSPACE, "--text", name, "--format", "json"])
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    matches = [item for item in experiments if item.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple Beaker experiments already use guarded name {name}")
    return str(matches[0]["id"]) if matches else None


def experiment_name(record: dict[str, Any], *, finalize_only: bool) -> str:
    variant = "original" if record["variant"] == "Original" else "dr-wt-embwd"
    suffix = "prune-all-post-finalizer-v1" if finalize_only else "all-post-e8-v1"
    return (
        f"dense-1b-bs{record['batchSequences']}-{variant}-lr{record['lr']}-"
        f"wd{record['wd']}-{suffix}"
    )


def create_experiment(
    record: dict[str, Any], revision: str, priority: str, *, finalize_only: bool
) -> str:
    name = experiment_name(record, finalize_only=finalize_only)
    existing = existing_named_experiment(name)
    if existing:
        return existing
    spec = spec_for(
        base_experiment=str(record["sourceExperiment"]),
        manifest=str(record["manifest"]),
        lr=str(record["lr"]),
        wd=str(record["wd"]),
        revision=revision,
        priority=priority,
        finalize_only=finalize_only,
        description=(
            f"Dense-1B BS{record['batchSequences']} {record['variant']} LR{record['lr']} "
            f"WD{record['wd']} all-POST-from-E8 coordinate; one node/eight GPUs."
        ),
    )
    output = run(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("Beaker submission succeeded without a parsed experiment ID")
    return identifiers[0]


def record_submission(
    report: dict[str, Any],
    record: dict[str, Any],
    experiment: str,
    revision: str,
    *,
    finalizer: bool,
) -> None:
    old = record.get("experiment")
    if old and old != experiment:
        history = record.setdefault("attemptHistory", [])
        if not any(item.get("experiment") == old for item in history):
            history.append(
                {
                    "experiment": old,
                    "reason": "stopped_after_higher_wd_won_same_epoch",
                    "replacedBy": experiment,
                }
            )
    record.update(
        {
            "status": "finalizing_pruned" if finalizer else "submitted",
            "beakerStatus": "submitted",
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "activeEpoch": record.get("activeEpoch") or 1,
        }
    )
    if finalizer:
        record["activePhase"] = "post_decay_finalize"
    batch = int(record["batchSequences"])
    chain_identifier = (
        "dense-1b-bs512-lr2e-3-conditional-followup" if batch == 512 else chain_id(batch)
    )
    chain = next(item for item in report["drWtEmbedWdGridChains"] if item["id"] == chain_identifier)
    chain.setdefault("experimentsByCoordinate", {})[record["id"]] = experiment
    if chain.get("status") in {"planned", "held", "conditional_held"}:
        chain["status"] = "submitted"


def primary_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    identifiers = {
        run_id(batch, lr, wd)
        for batch, config in PRIMARY.items()
        for lr, wd in config["coordinates"]
    }
    records = [record for record in report["runs"] if record.get("id") in identifiers]
    if len(records) != 6:
        raise RuntimeError(f"expected six primary Dense-1B coordinates, found {len(records)}")
    return records


def conditional_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    identifiers = {item["runId"] for item in CONDITIONAL}
    records = [record for record in report["runs"] if record.get("id") in identifiers]
    if len(records) != 3:
        raise RuntimeError(f"expected three conditional BS512 coordinates, found {len(records)}")
    return records


def release_primary(
    report: dict[str, Any], revision: str, priority: str, completed_count: int
) -> list[str]:
    if completed_count != TRIGGER_THRESHOLD:
        return [f"primary Dense-1B held: small={completed_count}/{TRIGGER_THRESHOLD}"]
    messages: list[str] = []
    for record in primary_records(report):
        if record.get("experiment"):
            messages.append(
                f"kept {record['id']} as {record['experiment']} ({record.get('status')})"
            )
            continue
        experiment = create_experiment(record, revision, priority, finalize_only=False)
        record_submission(report, record, experiment, revision, finalizer=False)
        messages.append(f"submitted {record['id']} as {experiment}")
    for batch in PRIMARY:
        chain = next(
            item for item in report["drWtEmbedWdGridChains"] if item["id"] == chain_id(batch)
        )
        if chain.get("status") in {"planned", "held"}:
            chain["status"] = "submitted"
        chain.update(
            {
                "completedSmallChainsAtSubmission": completed_count,
                "revision": revision,
                "reason": "All small chains confirmed POST saturation; independent coordinates released.",
            }
        )
    return messages


def release_conditional(report: dict[str, Any], revision: str, priority: str) -> list[str]:
    bs256 = next(item for item in report["drWtEmbedWdGridChains"] if item["id"] == chain_id(256))
    followup = next(
        item
        for item in report["drWtEmbedWdGridChains"]
        if item["id"] == "dense-1b-bs512-lr2e-3-conditional-followup"
    )
    if bs256.get("status") != "complete":
        return ["conditional BS512 held: BS256 is not terminal"]
    selected_lr = bs256.get("selectedLr")
    if selected_lr is None:
        raise RuntimeError("terminal BS256 chain has no selected LR")
    if Decimal(str(selected_lr)) != Decimal("2e-3"):
        followup.update(
            {
                "status": "not_required",
                "reason": f"Terminal BS256 winner used LR{selected_lr}, not LR2e-3.",
            }
        )
        for record in conditional_records(report):
            if not record.get("experiment"):
                record["status"] = "not_required"
        return [f"conditional BS512 not required: BS256 selected LR{selected_lr}"]
    messages: list[str] = []
    for record in conditional_records(report):
        if record.get("experiment"):
            messages.append(
                f"kept {record['id']} as {record['experiment']} ({record.get('status')})"
            )
            continue
        experiment = create_experiment(record, revision, priority, finalize_only=False)
        record_submission(report, record, experiment, revision, finalizer=False)
        messages.append(f"submitted {record['id']} as {experiment}")
    if followup.get("status") == "conditional_held":
        followup["status"] = "submitted"
    followup.update(
        {
            "revision": revision,
            "reason": "Released because the terminal BS256 winner used LR2e-3.",
        }
    )
    return messages


def finalize_pruned_run(
    report: dict[str, Any], identifier: str, revision: str, priority: str
) -> str:
    matches = [record for record in report["runs"] if record.get("id") == identifier]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate prune target {identifier}")
    record = matches[0]
    if not record.get("pruneRequested"):
        raise RuntimeError(f"{identifier} has no verified prune request")
    if len(record.get("postDecayResults", {})) < 3:
        raise RuntimeError(f"{identifier} cannot be pruned before three POST results")
    if record.get("status") == "pruned":
        return f"{identifier} already pruned"
    old = record.get("experiment")
    if not old:
        raise RuntimeError(f"{identifier} has no running experiment to replace")
    if not record.get("stopIssued"):
        run(["beaker", "experiment", "stop", str(old)])
        record["stopIssued"] = True
    experiment = create_experiment(record, revision, priority, finalize_only=True)
    record_submission(report, record, experiment, revision, finalizer=True)
    return f"{identifier}: stopped {old}; latest-three POST finalizer {experiment}"


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register-only", action="store_true")
    modes.add_argument("--submit-if-ready", action="store_true")
    modes.add_argument("--finalize-run")
    parser.add_argument("--revision")
    parser.add_argument("--priority", default="urgent")
    args = parser.parse_args()
    report = json.loads(REPORT_PATH.read_text())
    register_plan(report)
    completed_count, completed = successful_small_chains()
    for batch in PRIMARY:
        chain = next(
            item for item in report["drWtEmbedWdGridChains"] if item["id"] == chain_id(batch)
        )
        chain["completedSmallChainsAtLastCheck"] = completed_count
        chain["completedSmallChainIds"] = completed
        if completed_count < TRIGGER_THRESHOLD and not chain.get("experimentsByCoordinate"):
            chain["status"] = "held"
    if completed_count < TRIGGER_THRESHOLD:
        for record in primary_records(report):
            if not record.get("experiment"):
                record["status"] = "held"
    if args.register_only:
        write_report(report)
        print(f"registered Dense-1B all-POST-from-E8 plan; small={completed_count}/10")
        return
    if not args.revision:
        parser.error("--revision is required for submission/finalization")
    validate_revision(args.revision)
    if args.finalize_run:
        message = finalize_pruned_run(report, args.finalize_run, args.revision, args.priority)
        write_report(report)
        print(message)
        return
    messages = release_primary(report, args.revision, args.priority, completed_count)
    messages.extend(release_conditional(report, args.revision, args.priority))
    write_report(report)
    print("\n".join(messages))


if __name__ == "__main__":
    main()
