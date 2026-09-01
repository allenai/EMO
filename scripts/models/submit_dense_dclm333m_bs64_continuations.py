#!/usr/bin/env python3
"""Guardedly continue an authorized BS64 Pool-333M coordinate."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_dense_dclm333m_checkpoint_producer as runner
import submit_dense_dclm333m_checkpoint_producers as submitter

REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
REPORT_JS_PREFIX = "window.ICSL_CHECKPOINT_PRODUCER_GRID="


def selected_coordinates(
    config: dict[str, Any], coordinate_ids: list[str], target_epoch: int
) -> list[dict[str, Any]]:
    allowed = [
        item
        for item in config["producerCoordinates"]
        if target_epoch in runner.authorized_continuation_targets(item)
    ]
    if not coordinate_ids:
        return allowed
    selected = [item for item in allowed if item["id"] in coordinate_ids]
    if len(selected) != len(set(coordinate_ids)):
        raise RuntimeError(
            "every requested coordinate must be uniquely authorized for this continuation target"
        )
    return selected


def report_record(report: dict[str, Any], coordinate_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in report.get("dclm333mIntegratedRuns", [])
        if item.get("id") == coordinate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one report record for {coordinate_id}")
    return matches[0]


def validation_exact(record: dict[str, Any], epoch: int) -> float:
    result = record.get("postDecayResults", {}).get(str(epoch))
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise RuntimeError(f"{record['id']} is missing a healthy POST E{epoch} result")
    return float(result["validationExact"])


def ensure_ready(
    record: dict[str, Any], item: dict[str, Any], target_epoch: int
) -> None:
    source_epoch = runner.continuation_source_epoch(item, target_epoch)
    if source_epoch not in {
        int(epoch) for epoch in record.get("resolvedCheckpointEpochs", [])
    }:
        raise RuntimeError(f"{record['id']} is missing clean PD E{source_epoch}")
    previous_epochs = [
        int(epoch) for epoch in item["evaluationEpochs"] if int(epoch) < source_epoch
    ]
    if not previous_epochs:
        validation_exact(record, source_epoch)
        return
    previous_epoch = max(previous_epochs)
    previous = validation_exact(record, previous_epoch)
    current = validation_exact(record, source_epoch)
    if not current < previous:
        raise RuntimeError(
            f"{record['id']} saturated at E{source_epoch}: POST did not strictly "
            f"improve over E{previous_epoch}"
        )


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def register(
    report: dict[str, Any],
    item: dict[str, Any],
    experiment: str,
    revision: str,
    target_epoch: int,
) -> None:
    record = report_record(report, str(item["id"]))
    previous_experiment = str(record["experiment"])
    if previous_experiment != experiment:
        history = record.setdefault("experimentHistory", [])
        if not any(entry.get("experiment") == previous_experiment for entry in history):
            history.append(
                {
                    "experiment": previous_experiment,
                    "revision": record.get("revision"),
                    "status": record.get("status"),
                    "maxEpoch": max(record.get("retainedCheckpointEpochs", [32])),
                    "minRuntime": record.get("minRuntime"),
                }
            )
    source_epoch = runner.continuation_source_epoch(item, target_epoch)
    next_epochs = [
        int(epoch) for epoch in item["retainedCheckpointEpochs"] if int(epoch) > source_epoch
    ]
    if not next_epochs:
        raise RuntimeError(f"{item['id']} has no retained checkpoint after E{source_epoch}")
    record.update(
        {
            "experiment": experiment,
            "revision": revision,
            "retainedCheckpointEpochs": list(item["retainedCheckpointEpochs"]),
            "evaluationEpochs": list(item["evaluationEpochs"]),
            "continuationSourceEpoch": source_epoch,
            "continuationTargetEpoch": target_epoch,
            "minRuntime": "omitted",
            "status": "submitted",
            "beakerStatus": "submitted",
            "currentPhase": "producer",
            "currentEpoch": min(next_epochs),
        }
    )
    record.pop("job", None)
    record.pop("jobs", None)
    record.pop("wandbHealth", None)


def write_report(report: dict[str, Any]) -> None:
    report["updatedAt"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(report, indent=2) + "\n"
    atomic_text(REPORT, rendered)
    atomic_text(REPORT_JS, REPORT_JS_PREFIX + rendered.rstrip() + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-epoch", type=int, choices=runner.ALL_CONTINUATION_TARGETS, required=True
    )
    parser.add_argument("--coordinate", action="append", default=[])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    if args.print_specs == args.submit_if_ready:
        raise SystemExit("select exactly one of --print-specs or --submit-if-ready")

    submitter.validate_revision(args.revision)
    config = submitter.load_manifest()
    report = json.loads(REPORT.read_text())
    selected = selected_coordinates(config, args.coordinate, args.target_epoch)
    created: list[tuple[dict[str, Any], str]] = []
    for base_item in selected:
        item = runner.coordinate_for_target(config, str(base_item["id"]), args.target_epoch)
        record = report_record(report, str(item["id"]))
        ensure_ready(record, item, args.target_epoch)
        if args.print_specs:
            print(
                json.dumps(
                    submitter.spec_for(
                        item,
                        args.revision,
                        args.priority,
                        target_epoch=args.target_epoch,
                        omit_min_runtime=True,
                    ),
                    indent=2,
                )
            )
            continue
        experiment = submitter.create(
            item,
            args.revision,
            args.priority,
            target_epoch=args.target_epoch,
            omit_min_runtime=True,
        )
        created.append((item, experiment))
        print(f"{item['id']}: {experiment}")

    if created:
        for item, experiment in created:
            register(report, item, experiment, args.revision, args.target_epoch)
        write_report(report)


if __name__ == "__main__":
    main()
