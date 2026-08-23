#!/usr/bin/env python3
"""Stop the named small chains at exact boundaries and launch requested POST work."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORTS = {
    "474m": Path("reports/0802/data/wsd_batch_size_474m.json"),
    "153m": Path("reports/0802/data/wsd_batch_size_153m.json"),
}
FINALIZER_POLICY = "locked_wd_requested_postdecay_finalizer_v1"
PLAN: dict[tuple[str, int], dict[str, Any]] = {
    ("153m", 512): {
        "awaitPhase": "post_decay",
        "awaitEpoch": 128,
        "epochs": [],
        "stopOnSaturation": False,
    },
    ("153m", 256): {
        "awaitPhase": "post_decay",
        "awaitEpoch": 144,
        "epochs": [64, 80, 96],
        "stopOnSaturation": False,
    },
    ("153m", 128): {
        "awaitPhase": "post_decay",
        "awaitEpoch": 160,
        "epochs": [80, 96, 112, 128],
        "stopOnSaturation": False,
    },
    ("153m", 64): {
        "awaitPhase": "pre_decay",
        "awaitEpoch": 160,
        "epochs": [8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160],
        "stopOnSaturation": True,
    },
    ("474m", 256): {
        "awaitPhase": None,
        "awaitEpoch": None,
        "epochs": [48],
        "stopOnSaturation": False,
        "preserveExistingSelection": True,
    },
    ("474m", 128): {
        "awaitPhase": "pre_decay",
        "awaitEpoch": 88,
        "epochs": [64, 72, 80, 88],
        "stopOnSaturation": True,
    },
    ("474m", 64): {
        "awaitPhase": "pre_decay",
        "awaitEpoch": 88,
        "epochs": [40, 48, 56, 64, 72, 80, 88],
        "stopOnSaturation": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--submit-if-ready", action="store_true")
    return parser.parse_args()


def result_complete(record: dict[str, Any], plan: dict[str, Any]) -> bool:
    phase = plan["awaitPhase"]
    epoch = plan["awaitEpoch"]
    if phase is None:
        return True
    key = "postDecayResults" if phase == "post_decay" else "preDecayResults"
    return (record.get(key, {}).get(str(epoch)) or {}).get("status") == "complete"


def command_for(
    model: str,
    record: dict[str, Any],
    plan: dict[str, Any],
    revision: str,
) -> list[str]:
    completed = set(map(int, record.get("postDecayResults", {})))
    epochs = [epoch for epoch in plan["epochs"] if epoch not in completed]
    command = [
        ".venv/bin/python",
        "scripts/models/submit_small_dense_requested_postdecay_finalizer.py",
        "--model",
        model,
        "--global-sequences",
        str(record["batchSequences"]),
        "--revision",
        revision,
        "--resume-experiment",
        str(record["experiment"]),
        "--epochs",
        ",".join(map(str, epochs)),
        "--register",
    ]
    if plan.get("stopOnSaturation"):
        command.append("--stop-on-saturation")
    if plan.get("preserveExistingSelection"):
        command.append("--preserve-existing-selection")
    return command


def write_report(path: Path, report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise SystemExit("--revision must be the full 40-character pushed revision")
    summaries: list[str] = []
    for model, path in REPORTS.items():
        report = json.loads(path.read_text())
        changed = False
        records = {
            int(record["batchSequences"]): record
            for record in report.get("adaptiveDrWtEmbedWdChains", [])
        }
        for (planned_model, batch), plan in PLAN.items():
            if planned_model != model:
                continue
            record = records[batch]
            if record.get("policy") == FINALIZER_POLICY:
                summaries.append(
                    f"{model} BS{batch}: finalizer {record['experiment']} "
                    f"({record.get('status')})"
                )
                continue
            transition = dict(record.get("requestedPostDecayFinalization", {}))
            if not transition:
                transition = {
                    "requestedAt": datetime.now(tz=UTC).isoformat(),
                    "status": "awaiting-requested-boundary",
                    "oldExperiment": record["experiment"],
                    "lockedWd": str(record["lockedWd"]),
                    "awaitPhase": plan["awaitPhase"],
                    "awaitEpoch": plan["awaitEpoch"],
                    "requestedPostDecayEpochs": plan["epochs"],
                    "stopOnPostDecaySaturation": bool(plan.get("stopOnSaturation")),
                    "preserveExistingSelection": bool(
                        plan.get("preserveExistingSelection")
                    ),
                }
                record["requestedPostDecayFinalization"] = transition
                changed = True
            if not result_complete(record, plan):
                summaries.append(
                    f"{model} BS{batch}: waiting for {plan['awaitPhase']} "
                    f"E{plan['awaitEpoch']} in {record['experiment']}"
                )
                continue
            command = command_for(model, record, plan, args.revision)
            if args.submit_if_ready:
                if changed:
                    write_report(path, report)
                    changed = False
                subprocess.run(command, check=True)
                summaries.append(
                    f"{model} BS{batch}: stopped {record['experiment']} and submitted finalizer"
                )
                report = json.loads(path.read_text())
                records = {
                    int(item["batchSequences"]): item
                    for item in report.get("adaptiveDrWtEmbedWdChains", [])
                }
            else:
                summaries.append(f"{model} BS{batch}: READY " + " ".join(command))
        if changed:
            write_report(path, report)
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
