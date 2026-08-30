#!/usr/bin/env python3
"""Validate the split producer/evaluator manifest and its isolated report."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MANIFEST = Path("scripts/models/manifests/dense-constant-checkpoint-producers-v1.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")


def main() -> None:
    config = json.loads(MANIFEST.read_text())
    producers = config.get("producerCoordinates", [])
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
    for item in producers:
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
    if REPORT.is_file():
        report = json.loads(REPORT.read_text())
        if report.get("policy") != config["policy"]:
            raise RuntimeError("report and manifest policies differ")
        if [item["id"] for item in report.get("producers", [])] != ids:
            raise RuntimeError("report producer coordinates differ from the manifest")
        if [item["id"] for item in report.get("evaluators", [])] != [
            item["id"] for item in evaluators
        ]:
            raise RuntimeError("report evaluator coordinates differ from the manifest")
        prefix = "window.ICSL_CHECKPOINT_PRODUCER_GRID="
        text = REPORT_JS.read_text()
        if not text.startswith(prefix) or not text.endswith(";\n"):
            raise RuntimeError("report JS wrapper is malformed")
        if json.loads(text[len(prefix) : -2]) != report:
            raise RuntimeError("report JSON/JS mirrors differ")
    print("validated 10 checkpoint producers and 2 Dense-1B evaluators")


if __name__ == "__main__":
    main()
