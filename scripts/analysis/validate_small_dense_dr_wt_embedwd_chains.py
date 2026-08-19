#!/usr/bin/env python3
"""Validate the registered small-dense DR+WT+EmbedWD adaptive chains."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

MODELS = ("474m", "153m")
BATCHES = (64, 128, 256, 512)
ROOT = Path(__file__).resolve().parents[2]


def validate_model(model: str) -> None:
    path = ROOT / "reports" / "0802" / "data" / f"wsd_batch_size_{model}.json"
    report = json.loads(path.read_text())
    records = report.get("adaptiveDrWtEmbedWdChains", [])
    if len(records) != 4:
        raise ValueError(f"{model}: expected four adaptive chains, found {len(records)}")
    by_batch = {int(record["batchSequences"]): record for record in records}
    if sorted(by_batch) != list(BATCHES):
        raise ValueError(f"{model}: adaptive chain batches are {sorted(by_batch)}")
    experiments: set[str] = set()
    for batch, record in by_batch.items():
        expected_gpus = 4 if batch == 64 else 8
        expected_accumulation = batch // (expected_gpus * 16)
        expected_targets = (
            [1, 2, 4, 8, 16, 24]
            if model == "474m"
            else [1, 2, 4, 8, 16, 32, 48]
        )
        expected_increment = 8 if model == "474m" else 16
        checks = {
            "variant": record.get("variant") == "DR+WT+EmbedWD",
            "lr": str(record.get("lr")) == "2e-3",
            "rank microbatch": record.get("rankMicrobatchSequences") == 16,
            "GPU count": record.get("gpuCount") == expected_gpus,
            "gradient accumulation": record.get("gradientAccumulation") == expected_accumulation,
            "dynamic repacking": record.get("dynamicRepacking") is True,
            "data order": record.get("dataOrder") == "dynamic_repacking_from_e2",
            "weight tying": record.get("weightTying") is True,
            "embedding WD": record.get("decayEmbeddings") is True,
            "targets": record.get("initialTargets") == expected_targets,
            "increment": record.get("epochIncrement") == expected_increment,
            "retry count": record.get("automaticTaskRetries") == 8,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise ValueError(f"{model} BS{batch}: invalid {', '.join(failed)}")
        ladder = [str(value) for value in record["wdLadder"]]
        if not ladder or max(Decimal(value) for value in ladder) > Decimal("1.0"):
            raise ValueError(f"{model} BS{batch}: WD ladder exceeds the 1.0 cap")
        active_wds = [Decimal(str(value)) for value in record.get("activeWds", [])]
        if active_wds and max(active_wds) > Decimal("1.0"):
            raise ValueError(f"{model} BS{batch}: active WD exceeds the 1.0 cap")
        for epoch, results in record.get("results", {}).items():
            if any(Decimal(str(wd)) > Decimal("1.0") for wd in results):
                raise ValueError(f"{model} BS{batch} E{epoch}: result WD exceeds 1.0")
        center = str(record["baselineOptimalWd"])
        center_index = ladder.index(center)
        if record.get("initialWds") != ladder[center_index - 1 : center_index + 2]:
            raise ValueError(f"{model} BS{batch}: initial WD neighbors are invalid")
        outputs = record.get("outputByWd", {})
        if sorted(outputs) != sorted(ladder):
            raise ValueError(f"{model} BS{batch}: output map does not cover the WD ladder")
        if len(set(outputs.values())) != len(ladder):
            raise ValueError(f"{model} BS{batch}: WD trajectories share an output directory")
        for wd, output in outputs.items():
            expected_suffix = f"/bs{batch}_dr_wt_embwd_lr2e-3_wd{wd}"
            if not str(output).endswith(expected_suffix):
                raise ValueError(f"{model} BS{batch}: noncanonical output {output}")
        experiment = str(record.get("experiment", ""))
        if len(experiment) != 26:
            raise ValueError(f"{model} BS{batch}: missing Beaker experiment")
        if experiment in experiments:
            raise ValueError(f"{model}: duplicate Beaker experiment {experiment}")
        experiments.add(experiment)


def main() -> None:
    for model in MODELS:
        validate_model(model)
    print("small-dense DR+WT+EmbedWD chains validated")


if __name__ == "__main__":
    main()
