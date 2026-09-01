#!/usr/bin/env python3
"""Validate the mixed 1B-v1 and corrected small-model Pool-3B-v2 grid."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MANIFEST = Path("scripts/models/manifests/dense-constant-checkpoint-producers-v1.json")
SMALL_POOL3B_MANIFEST = Path(
    "scripts/models/manifests/dense-small-pool3b-checkpoint-producers-v2.json"
)
DCLM333M_MANIFEST = Path("scripts/models/manifests/dense-dclm333m-checkpoint-producers-v1.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
REPORT_POLICY = "dense_constant_checkpoint_producers_v1_pool3b_small_v2"


def main() -> None:
    config = json.loads(MANIFEST.read_text())
    small_config = json.loads(SMALL_POOL3B_MANIFEST.read_text())
    dclm333m_config = json.loads(DCLM333M_MANIFEST.read_text())
    producers = [
        item for item in config.get("producerCoordinates", []) if item["model"] == "1b"
    ] + small_config.get("producerCoordinates", [])
    evaluators = config.get("evaluatorCoordinates", [])
    if len(producers) != 10 or len(evaluators) != 2:
        raise RuntimeError("expected exactly ten producers and two 1B evaluators")
    ids = [item["id"] for item in producers]
    if len(ids) != len(set(ids)):
        raise RuntimeError("producer IDs are not unique")
    expected_models = {"1b": 2, "474m": 4, "153m": 4}
    actual_models = {
        model: sum(item["model"] == model for item in producers) for model in expected_models
    }
    if actual_models != expected_models:
        raise RuntimeError(f"producer model counts differ: {actual_models}")
    for item in producers[:2]:
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/models/run_dense_constant_checkpoint_producer.py",
                "--manifest",
                str(MANIFEST),
                "--coordinate",
                item["id"],
                "--validate-only",
            ],
            check=True,
        )
    for item in producers[2:]:
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/models/run_dense_small_pool3b_checkpoint_producer.py",
                "--manifest",
                str(SMALL_POOL3B_MANIFEST),
                "--coordinate",
                item["id"],
                "--validate-only",
            ],
            check=True,
        )
    for item in evaluators:
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/models/run_dense_1b_checkpoint_evaluator.py",
                "--manifest",
                str(MANIFEST),
                "--evaluator",
                item["id"],
                "--validate-only",
            ],
            check=True,
        )
    dclm333m_runs = dclm333m_config.get("producerCoordinates", [])
    if len(dclm333m_runs) != 12:
        raise RuntimeError("expected exactly twelve Pool-333M integrated runs")
    for item in dclm333m_runs:
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/models/run_dense_dclm333m_checkpoint_producer.py",
                "--manifest",
                str(DCLM333M_MANIFEST),
                "--coordinate",
                item["id"],
                "--validate-only",
            ],
            check=True,
        )
    if REPORT.is_file():
        report = json.loads(REPORT.read_text())
        if report.get("policy") != REPORT_POLICY:
            raise RuntimeError("report and manifest policies differ")
        if [item["id"] for item in report.get("producers", [])] != ids:
            raise RuntimeError("report producer coordinates differ from the manifest")
        if [item["id"] for item in report.get("evaluators", [])] != [
            item["id"] for item in evaluators
        ]:
            raise RuntimeError("report evaluator coordinates differ from the manifest")
        if [item["id"] for item in report.get("dclm333mIntegratedRuns", [])] != [
            item["id"] for item in dclm333m_runs
        ]:
            raise RuntimeError("Pool-333M report coordinates differ from the manifest")
        if int(report.get("dclm333mIntegratedCount", -1)) != len(dclm333m_runs):
            raise RuntimeError("Pool-333M report count is stale")
        prefix = "window.ICSL_CHECKPOINT_PRODUCER_GRID="
        text = REPORT_JS.read_text()
        if not text.startswith(prefix) or not text.endswith(";\n"):
            raise RuntimeError("report JS wrapper is malformed")
        if json.loads(text[len(prefix) : -2]) != report:
            raise RuntimeError("report JSON/JS mirrors differ")
    print(
        "validated 2 Dense-1B v1 producers, 8 Pool-3B v2 producers, "
        "12 Pool-333M integrated runs, and 2 evaluators"
    )


if __name__ == "__main__":
    main()
