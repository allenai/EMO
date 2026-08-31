#!/usr/bin/env python3
"""Validate registered standalone small-model Pool-3B evaluator records."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(SCRIPT_DIR))
import evaluator_min_runtime
import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer

MANIFEST = Path("scripts/models/manifests/dense-small-pool3b-checkpoint-producers-v2.json")
REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")


def main() -> None:
    config = producer.load_manifest(MANIFEST)
    coordinates = {item["id"]: item for item in config["producerCoordinates"]}
    report = json.loads(REPORT.read_text())
    records = report.get("smallEvaluators", [])
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("small evaluator IDs are not unique")
    for record in records:
        producer_id = str(record["producerId"])
        if producer_id not in coordinates:
            raise RuntimeError(f"unknown small evaluator producer {producer_id}")
        item = coordinates[producer_id]
        epoch = int(record["epoch"])
        expected_id = f"{producer_id}-post-e{epoch}-v1"
        if record["id"] != expected_id:
            raise RuntimeError(f"small evaluator ID mismatch: {record['id']}")
        if record.get("policy") != evaluator.POLICY:
            raise RuntimeError(f"small evaluator policy mismatch: {record['id']}")
        if record.get("sourceCheckpoint") != str(evaluator.source_checkpoint(item, epoch)):
            raise RuntimeError(f"small evaluator source mismatch: {record['id']}")
        expected_runtime = evaluator_min_runtime.estimate_min_runtime(
            model=str(item["model"]),
            pool_tokens=producer.TARGET_POOL_TOKENS,
            batch_sequences=int(item["batchSequences"]),
            epochs=[epoch],
        )["minRuntime"]
        if record.get("minRuntime") != expected_runtime:
            raise RuntimeError(f"small evaluator minRuntime mismatch: {record['id']}")
        if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", str(record.get("experiment", ""))):
            raise RuntimeError(f"small evaluator experiment ID malformed: {record['id']}")
        result = record.get("postDecayResult")
        if result is not None and (
            result.get("comparisonGroup") != "post_decay" or int(result.get("epoch", -1)) != epoch
        ):
            raise RuntimeError(f"small evaluator result provenance mismatch: {record['id']}")
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/models/run_dense_small_pool3b_checkpoint_evaluator.py",
                "--manifest",
                str(MANIFEST),
                "--coordinate",
                producer_id,
                "--epoch",
                str(epoch),
                "--validate-only",
            ],
            check=True,
        )
    prefix = "window.ICSL_CHECKPOINT_PRODUCER_GRID="
    text = REPORT_JS.read_text()
    if not text.startswith(prefix) or not text.endswith(";\n"):
        raise RuntimeError("report JS wrapper is malformed")
    if json.loads(text[len(prefix) : -2]) != report:
        raise RuntimeError("report JSON/JS mirrors differ")
    print(f"validated {len(records)} standalone small-model evaluators")


if __name__ == "__main__":
    main()
