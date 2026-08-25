#!/usr/bin/env python3
"""Resume user-stopped 153M chains under the POST-only policy."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_small_dense_postdecay_only_continuation as runner
import submit_small_dense_dr_wt_embedwd_chain as submit

POLICY = runner.POLICY
REPORT_PATH = Path("reports/0802/data/wsd_batch_size_153m.json")
TARGETS = {64: 160, 128: 160, 256: 144, 512: 128}
MANIFEST_TEMPLATE = (
    "scripts/models/manifests/"
    "small-dense-153m-bs{batch}-dr-wt-embwd-postdecay-only-continuation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--write-manifests-only", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    parser.add_argument("--workspace", default=submit.WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    return parser.parse_args()


def record_by_batch(report: dict[str, Any], batch: int) -> dict[str, Any]:
    matches = [
        record
        for record in report.get("adaptiveDrWtEmbedWdChains", [])
        if int(record.get("batchSequences", 0)) == batch
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one registered 153M BS{batch} chain")
    return matches[0]


def reduced_post_result(epoch: int, result: dict[str, Any], wd: str) -> dict[str, Any]:
    if result.get("status") != "complete":
        raise RuntimeError(f"E{epoch} POST result is not complete")
    if result.get("comparisonGroup") != "post_decay":
        raise RuntimeError(f"E{epoch} is not a POST result")
    return {
        "epoch": epoch,
        "status": "complete",
        "phase": "post_decay",
        "comparisonGroup": "post_decay",
        "validation": result.get("validation"),
        "validationExact": result["validationExact"],
        "checkpoint": result["checkpoint"],
        "sourcePreDecayCheckpoint": result.get("sourcePreDecayCheckpoint"),
        "wd": wd,
        "variant": "DR+WT+EmbedWD",
    }


def manifest_from_record(record: dict[str, Any], boundary: int) -> dict[str, Any]:
    batch = int(record["batchSequences"])
    wd = str(record["lockedWd"])
    if record.get("postDecaySaturated") is not False:
        raise RuntimeError(f"BS{batch} is not registered as POST-unsaturated")
    selection = record.get("postDecaySelection") or {}
    if selection.get("status") != "stopped_by_user":
        raise RuntimeError(f"BS{batch} is not a user-stopped chain")
    completed = {
        int(epoch): result
        for epoch, result in record.get("postDecayResults", {}).items()
        if int(epoch) <= boundary and result.get("status") == "complete"
    }
    if len(completed) < runner.POST_DECAY_SOURCE_COUNT or boundary not in completed:
        raise RuntimeError(f"BS{batch} lacks complete POST provenance through E{boundary}")
    source = completed[boundary].get("sourcePreDecayCheckpoint")
    if not source:
        raise RuntimeError(f"BS{batch} E{boundary} lacks its source checkpoint")
    expected_step = runner.adaptive.stable_step(boundary, batch)
    if Path(str(source)).name != f"step{expected_step}":
        raise RuntimeError(f"BS{batch} E{boundary} source step is not exact")
    output_root = submit.output_root("153m")
    continuation_root = (
        f"{output_root}/.bs{batch}_dr_wt_embwd_lr{record['lr']}_wd{wd}_"
        "postdecay_only_continuation_v1"
    )
    prior = {
        str(epoch): reduced_post_result(epoch, completed[epoch], wd)
        for epoch in sorted(completed)
    }
    manifest = {
        "model": "153m",
        "baseExperiment": submit.MODELS["153m"]["baseExperiment"],
        "globalSequences": batch,
        "nprocPerNode": submit.nproc_for(batch),
        "rankMicrobatchSequences": int(record["rankMicrobatchSequences"]),
        "gradientAccumulation": int(record["gradientAccumulation"]),
        "warmupSteps": submit.warmup_steps(batch),
        "learningRate": str(record["lr"]),
        "wdLadder": [str(value) for value in record["wdLadder"]],
        "initialTargets": [int(value) for value in record["initialTargets"]],
        "epochIncrement": int(record["epochIncrement"]),
        "outputRoot": output_root,
        "runSuffix": "sm0824-postonly-continuation",
        "variant": "DR+WT+EmbedWD",
        "dataOrder": "dynamic_repacking_from_e2",
        "weightTying": True,
        "decayEmbeddings": True,
        "policy": POLICY,
        "lockedWd": wd,
        "postDecaySourceCount": runner.POST_DECAY_SOURCE_COUNT,
        "comparisonPolicy": "post_decay_only",
        "preDecayEvaluation": False,
        "postDecaySaturationCriterion": "strict_non_improvement",
        "resumeEpoch": boundary,
        "resumeCheckpoint": str(source),
        "continuationRoot": continuation_root,
        "priorPostDecayResults": prior,
    }
    runner.validate_config(manifest)
    return manifest


def manifest_path(batch: int) -> Path:
    return Path(MANIFEST_TEMPLATE.format(batch=batch))


def write_manifests(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifests: dict[int, dict[str, Any]] = {}
    for batch, boundary in TARGETS.items():
        record = record_by_batch(report, batch)
        if record.get("policy") == POLICY:
            path = manifest_path(batch)
            manifest = json.loads(path.read_text())
            runner.validate_config(manifest)
        else:
            manifest = manifest_from_record(record, boundary)
            path = manifest_path(batch)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifests[batch] = manifest
    return manifests


def validate_revision(revision: str | None) -> str:
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SystemExit("--revision must be a full 40-character pushed revision")
    submit.run(["git", "merge-base", "--is-ancestor", revision, "origin/sewonm/icsl"])
    return revision


def experiment_name(batch: int) -> str:
    return f"dense-153m-bs{batch}-postdecay-only-continuation-v1"


def build_spec(
    batch: int,
    manifest: dict[str, Any],
    revision: str,
    *,
    priority: str,
) -> dict[str, Any]:
    base = submit.base_spec(str(manifest["baseExperiment"]))
    submit.audit_base_spec("153m", base)
    spec = copy.deepcopy(base)
    task = spec["tasks"][0]
    task["arguments"] = [
        "python",
        "scripts/models/run_small_dense_postdecay_only_continuation.py",
        "--manifest",
        str(manifest_path(batch)),
    ]
    task["envVars"] = [
        variable
        for variable in task.get("envVars", [])
        if variable.get("name") not in {"GANTRY_USE_TORCHRUN", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    submit.set_revision(task, revision)
    task["resources"] = {
        "gpuCount": int(manifest["nprocPerNode"]),
        "sharedMemory": "10 GiB",
    }
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": True}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Dense 153M BS{batch} locked-WD{manifest['lockedWd']} POST-only continuation "
        f"from exact E{manifest['resumeEpoch']}. Constant-LR frontiers are checkpoint-only; "
        "every new frontier is independently WSD-decayed/evaluated until POST saturation."
    )
    return spec


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_PATH.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def register(
    report: dict[str, Any],
    record: dict[str, Any],
    manifest: dict[str, Any],
    revision: str,
    experiment: str,
) -> None:
    batch = int(record["batchSequences"])
    previous_experiment = str(record["experiment"])
    record.setdefault("attemptHistory", []).append(
        {
            "beaker": previous_experiment,
            "status": "user-stopped-unsaturated-superseded-by-post-only-continuation",
            "activePhase": record.get("activePhase"),
            "activeEpoch": record.get("activeEpoch"),
            "wandbHealth": record.get("wandbHealth"),
            "replacedBy": experiment,
        }
    )
    previous_selection = record.pop("postDecaySelection", None)
    if previous_selection is not None:
        record["supersededUserStopSelection"] = previous_selection
    previous_finalization = record.pop("requestedPostDecayFinalization", None)
    if previous_finalization is not None:
        record["supersededRequestedPostDecayFinalization"] = previous_finalization
    previous_result = record.pop("requestedPostDecayFinalizationResult", None)
    if previous_result is not None:
        record["supersededRequestedPostDecayFinalizationResult"] = previous_result
    manual = record.pop("manualNoWorkFinalization", None)
    if manual is not None:
        record["supersededManualNoWorkFinalization"] = manual
    next_frontier = runner.next_epoch(manifest, int(manifest["resumeEpoch"]))
    record.update(
        {
            "policy": POLICY,
            "status": "submitted",
            "beakerStatus": "submitted",
            "experiment": experiment,
            "beaker": experiment,
            "revision": revision,
            "recoveryOf": previous_experiment,
            "comparisonPolicy": "post_decay_only",
            "preDecayEvaluation": False,
            "wdTuningStopped": True,
            "postDecaySourceCount": runner.POST_DECAY_SOURCE_COUNT,
            "postDecaySaturationCriterion": "strict_non_improvement",
            "postDecaySaturated": False,
            "continuationBoundaryEpoch": int(manifest["resumeEpoch"]),
            "continuationSourceCheckpoint": manifest["resumeCheckpoint"],
            "continuationOutput": manifest["continuationRoot"],
            "continuationManifest": str(manifest_path(batch)),
            "activePhase": "postdecay_only_continuation_pending",
            "activeEpoch": next_frontier,
            "activeWds": [str(record["lockedWd"])],
            "reason": (
                f"The user reopened this still-improving chain from the exact E"
                f"{manifest['resumeEpoch']} constant-LR checkpoint. New frontiers are "
                "checkpoint-only and saturation compares POST with POST."
            ),
            "wandbHealth": {
                "status": "pending",
                "checkedAt": datetime.now(tz=UTC).isoformat(),
                "run": None,
                "url": None,
                "warnings": [],
                "criticalSignals": [],
                "shouldRecover": False,
                "reason": "The POST-only continuation has not emitted a stage start yet.",
            },
        }
    )
    for key in (
        "selectedPostDecayEpoch",
        "selectedPostDecayValidationExact",
        "selectedCheckpoint",
        "saturatedEpoch",
        "stopReason",
        "needsPolicyResume",
        "needsAttention",
        "progress",
    ):
        record.pop(key, None)
    write_report(report)


def main() -> None:
    args = parse_args()
    report = json.loads(REPORT_PATH.read_text())
    manifests = write_manifests(report)
    if args.write_manifests_only:
        for batch in TARGETS:
            print(f"wrote {manifest_path(batch)}")
        return
    revision = validate_revision(args.revision)
    ready: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for batch in TARGETS:
        record = record_by_batch(report, batch)
        if record.get("policy") == POLICY:
            print(f"BS{batch}: already registered as {record['experiment']}")
            continue
        expected = manifest_from_record(record, TARGETS[batch])
        if manifests[batch] != expected:
            raise RuntimeError(f"BS{batch} committed manifest no longer matches the report")
        ready.append((batch, record, manifests[batch]))
    if not args.submit_if_ready:
        for batch, record, manifest in ready:
            print(
                f"BS{batch}: READY from E{manifest['resumeEpoch']} WD{manifest['lockedWd']} "
                f"after {record['experiment']}"
            )
        return
    specs: dict[int, dict[str, Any]] = {}
    for batch, _, manifest in ready:
        name = experiment_name(batch)
        submit.audit_name(args.workspace, name)
        specs[batch] = build_spec(
            batch, manifest, revision, priority=args.priority
        )
    for batch, record, manifest in ready:
        output = submit.run(
            [
                "beaker",
                "experiment",
                "create",
                "-",
                "--name",
                experiment_name(batch),
                "--workspace",
                args.workspace,
            ],
            input_text=json.dumps(specs[batch]),
        )
        print(output, end="")
        identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
        if not identifiers:
            raise RuntimeError(f"BS{batch} submission has no parsed experiment ID")
        register(report, record, manifest, revision, identifiers[0])


if __name__ == "__main__":
    main()
