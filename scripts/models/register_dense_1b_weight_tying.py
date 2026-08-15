#!/usr/bin/env python3
"""Register the Dense-1B DR weight-tying comparison in the 0802 report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
TARGETS = [1, 2, 4, 8, 12, 16, 20, 24, 28]

COLUMNS = (
    {
        "key": "drwt64",
        "label": "BS64 · DR+WT",
        "batchSequences": 64,
        "dataOrder": "dynamic_repacking",
        "weightTying": True,
        "initialWd": "0.3",
        "color": "#a855f7",
    },
    {
        "key": "drwt512",
        "label": "BS512 · DR+WT",
        "batchSequences": 512,
        "dataOrder": "dynamic_repacking",
        "weightTying": True,
        "initialWd": "0.333",
        "color": "#7c3aed",
    },
)

RUNS = (
    {
        "id": "drwt64-lr1e-3-wd0.3",
        "method": "drwt64",
        "batchSequences": 64,
        "dataOrder": "dynamic_repacking",
        "weightTying": True,
        "lr": "1e-3",
        "wd": "0.3",
        "status": "planned",
        "activeEpoch": 1,
        "attemptedEpochs": [],
        "sourceExperiment": "01KZDCD9AYJ0REGZTE2YG18NHF",
        "sourceCheckpoint": "fresh",
        "gpuCount": 8,
        "nodeCount": 1,
        "rankMicrobatchSequences": 8,
        "gradientAccumulationSteps": 1,
        "plannedTargets": TARGETS,
        "results": {},
        "reason": "Fresh tied-weight E1 bootstrap; DR begins at E2 from its exact tied E1 pre-decay checkpoint.",
    },
    {
        "id": "drwt512-lr1e-3-wd0.333",
        "method": "drwt512",
        "batchSequences": 512,
        "dataOrder": "dynamic_repacking",
        "weightTying": True,
        "lr": "1e-3",
        "wd": "0.333",
        "status": "planned",
        "activeEpoch": 1,
        "attemptedEpochs": [],
        "sourceExperiment": "01KZFVTS3EHFR7HQ5ZY9DAF9NS",
        "sourceCheckpoint": "fresh",
        "gpuCount": 32,
        "nodeCount": 4,
        "rankMicrobatchSequences": 8,
        "gradientAccumulationSteps": 2,
        "plannedTargets": TARGETS,
        "results": {},
        "reason": "Fresh tied-weight E1 bootstrap; DR begins at E2 from its exact tied E1 pre-decay checkpoint.",
    },
    {
        "id": "drwt512-lr1e-3-wd1.0",
        "method": "drwt512",
        "batchSequences": 512,
        "dataOrder": "dynamic_repacking",
        "weightTying": True,
        "lr": "1e-3",
        "wd": "1.0",
        "status": "planned",
        "activeEpoch": 1,
        "attemptedEpochs": [],
        "sourceExperiment": "01KZGE4HMCEWMBP86YEAGSP1B7",
        "sourceCheckpoint": "fresh",
        "gpuCount": 32,
        "nodeCount": 4,
        "rankMicrobatchSequences": 8,
        "gradientAccumulationSteps": 2,
        "plannedTargets": TARGETS,
        "results": {},
        "reason": "Fresh tied-weight E1 bootstrap; DR begins at E2 from its exact tied E1 pre-decay checkpoint.",
    },
)


def upsert_after(items: list[dict[str, Any]], value: dict[str, Any], after_key: str) -> None:
    existing = next((item for item in items if item.get("key") == value["key"]), None)
    if existing is not None:
        existing.update(value)
        return
    index = next(i for i, item in enumerate(items) if item.get("key") == after_key)
    items.insert(index + 1, value.copy())


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    upsert_after(report["columns"], COLUMNS[0], "dr64")
    upsert_after(report["columns"], COLUMNS[1], "dr512")

    existing_runs = {run["id"]: run for run in report["runs"]}
    for definition in RUNS:
        if definition["id"] in existing_runs:
            # Preserve live state and results when this registration is rerun.
            for key, value in definition.items():
                if key not in {"status", "activeEpoch", "attemptedEpochs", "results", "reason"}:
                    existing_runs[definition["id"]][key] = value
        else:
            report["runs"].append(definition.copy())

    report["weightTyingStudy"] = {
        "model": "Dense 1B",
        "dataOrder": "dynamic_repacking_from_e2",
        "learningRate": "1e-3",
        "targets": TARGETS,
        "coordinates": [
            {"batchSequences": 64, "weightDecay": "0.3"},
            {"batchSequences": 512, "weightDecay": "0.333"},
            {"batchSequences": 512, "weightDecay": "1.0"},
        ],
        "advanceRule": (
            "Advance each coordinate from its exact pre-decay checkpoint while held-out "
            "DCLM validation CE strictly improves; retain the first non-improving endpoint "
            "as overfitting evidence and then stop that coordinate."
        ),
        "optimizerPolicy": (
            "The tied matrix remains the embeddings.weight optimizer group and therefore "
            "uses zero weight decay; the reported WD applies to all ordinary decayed groups."
        ),
    }
    setup_note = (
        " Weight-tying runs share the input embedding and LM-head projection as one parameter; "
        "DR+WT is displayed immediately after its matched DR column."
    )
    if "Weight-tying runs" not in report["setup"]:
        report["setup"] += setup_note
    selection_note = (
        " WT coordinates advance independently while validation CE strictly improves and stop "
        "after the first non-improving endpoint; train–validation gaps are compared against the "
        "same-batch, same-LR, same-WD DR trajectory."
    )
    if "WT coordinates advance" not in report["selection"]:
        report["selection"] += selection_note
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    write_report(report)


if __name__ == "__main__":
    main()
