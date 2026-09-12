#!/usr/bin/env python3
"""Submit the explicit 153M Pool-111M BS64 E32-to-E48 continuation."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import run_dense_dclm111m_checkpoint_producer as runner
import submit_dense_dclm111m_checkpoint_producers as registry_base
import submit_dense_dclm333m_checkpoint_producers as submit_base


COORDINATE = "dense-153m-dclm111m-bs64-lr2e-3-wd0.3"
TARGET_EPOCH = 48
SOURCE_EPOCH = 32
NAME = f"{COORDINATE}-continuation-e32-e48-allocated-v1"
MIN_RUNTIME = "2h"
REGISTRY = Path("reports/0802/data/wsd_pool111m_grid.json")
REGISTRY_JS = REGISTRY.with_suffix(".js")


def configure() -> None:
    runner.configure_base()
    submit_base.runner = runner
    submit_base.MANIFEST = registry_base.MANIFEST
    submit_base.RUNNER = registry_base.RUNNER
    submit_base.POOL_DISPLAY_NAME = runner.POOL_DISPLAY_NAME


def record_for(report: dict) -> dict:
    matches = [item for item in report["trajectories"] if item.get("id") == COORDINATE]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {COORDINATE} registry record")
    return matches[0]


def verify_ready(record: dict, item: dict) -> None:
    if record.get("status") in {"submitted", "queued", "scheduled", "running"}:
        raise RuntimeError("coordinate already has an active writer")
    resolved = {int(epoch) for epoch in record.get("resolvedCheckpointEpochs", [])}
    if SOURCE_EPOCH not in resolved:
        raise RuntimeError("exact E32 PD source is not registered as retained")
    expected_source = Path(str(record["output"])) / f"step{runner.stable_step(SOURCE_EPOCH, 64)}"
    if Path(str(item["output"])) != Path(str(record["output"])):
        raise RuntimeError("continuation output differs from the registered canonical output")
    if runner.continuation_source_epoch(item, TARGET_EPOCH) != SOURCE_EPOCH:
        raise RuntimeError(f"continuation does not resolve exact source {expected_source}")


def register(report: dict, experiment: str, revision: str, spec: dict) -> None:
    record = record_for(report)
    history = record.setdefault("experimentHistory", [])
    prior = {
        "experiment": record.get("experiment"),
        "job": record.get("job"),
        "revision": record.get("revision"),
        "status": record.get("status"),
        "terminalReason": record.get("terminalReason"),
        "maxValidatedEpoch": max(record.get("resolvedPostEpochs") or [SOURCE_EPOCH]),
        "output": record.get("output"),
    }
    if not any(entry.get("experiment") == prior["experiment"] for entry in history):
        history.append(prior)
    min_runtime = spec["tasks"][0]["context"].get("minRuntime")
    record.update(
        {
            "status": "submitted",
            "beakerStatus": "submitted",
            "experiment": experiment,
            "revision": revision,
            "currentEpoch": 40,
            "currentPhase": "producer",
            "continuationSourceEpoch": SOURCE_EPOCH,
            "continuationSourceCheckpoint": str(
                Path(str(record["output"])) / f"step{runner.stable_step(SOURCE_EPOCH, 64)}"
            ),
            "continuationTargetEpoch": TARGET_EPOCH,
            "continuationCheckpointEpochs": [40, 48],
            "continuationEvaluationEpochs": [48],
            "continuationIgnoresPriorE32Saturation": True,
            "continuationHardStopEpoch": TARGET_EPOCH,
            "submittedAt": datetime.now(tz=UTC).isoformat(),
            "minRuntime": min_runtime,
        }
    )
    for key in ("job", "jobs", "terminalReason", "completedAt", "wandbHealth"):
        record.pop(key, None)
    attempted_evaluator = report.get("retainedCheckpointEvaluation")
    if attempted_evaluator and attempted_evaluator.get("producerId") == COORDINATE:
        attempted_evaluator.update(
            {
                "status": "terminal_blocked_missing_sources_at_attempt_time",
                "supersededByContinuationExperiment": experiment,
            }
        )
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    rendered = json.dumps(report, indent=2) + "\n"
    registry_base.atomic_text(REGISTRY, rendered)
    registry_base.atomic_text(
        REGISTRY_JS, "window.ICSL_POOL111M_GRID=" + rendered.rstrip() + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    if args.print_spec == args.submit_if_ready:
        raise SystemExit("choose exactly one of --print-spec or --submit-if-ready")

    configure()
    submit_base.validate_revision(args.revision)
    manifest = runner.load_manifest(registry_base.MANIFEST)
    item = runner.coordinate_for_target(manifest, COORDINATE, TARGET_EPOCH)
    report = json.loads(REGISTRY.read_text())
    verify_ready(record_for(report), item)
    spec = submit_base.spec_for(
        item,
        args.revision,
        args.priority,
        target_epoch=TARGET_EPOCH,
        omit_min_runtime=False,
    )
    spec["tasks"][0]["context"]["minRuntime"] = MIN_RUNTIME
    spec["description"] = (
        "Explicitly authorized 153M DCLM-111M BS64 LR2e-3 WD0.3 continuation "
        "from exact retained E32 PD through E40 recovery PD and exact E48 PD, "
        "then isolated uncapped 10% WSD decay plus heldout/downstream evaluation "
        "at E48 and hard stop. One 4-H100 node, rank microbatch 16, GA1, "
        "canonical output, forced exact source, auto-resume and eight retries; "
        f"allocated with minRuntime={MIN_RUNTIME}."
    )
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return

    output = submit_base.command(
        [
            "beaker", "experiment", "create", "-", "--name", NAME,
            "--workspace", submit_base.WORKSPACE,
        ],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission returned no experiment ID")
    register(report, identifiers[0], args.revision, spec)
    print(f"{NAME}: {identifiers[0]}")


if __name__ == "__main__":
    main()
