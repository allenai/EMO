#!/usr/bin/env python3
"""Refresh nested-3B stopping evidence from the live trusted 1B reports."""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "0802" / "data"
TARGETS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88)
MODELS = {
    "153m": "2e-3",
    "474m": "2e-3",
    "1b": "1e-3",
}
BATCHES = (64, 128, 256)


def trusted_optimum(source: dict, batch: int, lr: str) -> tuple[int, Decimal]:
    unhealthy = set(source.get("healthAudit", {}).get("unhealthy", {}))
    rows: list[tuple[Decimal, int]] = []
    epochs = {
        int(epoch)
        for sweep in source.get("batchSweeps", [])
        if sweep.get("batchSequences") == batch
        and Decimal(str(sweep.get("lr"))) == Decimal(lr)
        for epoch in sweep.get("results", {})
        if str(epoch).isdigit() and int(epoch) >= 1
    }
    for epoch in epochs:
        candidates: list[tuple[Decimal, Decimal]] = []
        for sweep in source.get("batchSweeps", []):
            if (
                sweep.get("batchSequences") != batch
                or Decimal(str(sweep.get("lr"))) != Decimal(lr)
            ):
                continue
            result = sweep.get("results", {}).get(str(epoch), {})
            if (
                result.get("status") != "complete"
                or result.get("validation") is None
                or result.get("wandb") in unhealthy
            ):
                continue
            candidates.append(
                (Decimal(str(result["validation"])), Decimal(str(sweep["wd"])))
            )
        if candidates:
            rows.append((min(candidates)[0], epoch))
    if not rows:
        raise RuntimeError(f"no trusted 1B-pool optimum for BS{batch} at LR {lr}")
    validation, epoch = min(rows)
    return epoch, validation


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


def main() -> None:
    for model, lr in MODELS.items():
        source_path = DATA / f"wsd_batch_size_{model}.json"
        report_path = DATA / f"wsd_batch_size_{model}_pool3b.json"
        source = json.loads(source_path.read_text())
        report = json.loads(report_path.read_text())
        optimums: dict[str, int] = {}
        ceilings: dict[str, int] = {}
        for batch in BATCHES:
            optimum, _ = trusted_optimum(source, batch, lr)
            ceiling = next((target for target in TARGETS if target >= optimum), TARGETS[-1])
            optimums[str(batch)] = optimum
            ceilings[str(batch)] = ceiling
        report["baseline1b"] = {
            "source": str(source_path.relative_to(ROOT)),
            "batchSweeps": deepcopy(source.get("batchSweeps", [])),
            "healthAudit": deepcopy(source.get("healthAudit", {})),
            "trustedOptimalEpochByBatch": optimums,
            "stopCeilingByBatch": ceilings,
        }
        report["poolPlan"]["sourceOptimalEpochByBatch"] = optimums
        report["poolPlan"]["stopCeilingByBatch"] = ceilings
        report["batchTargetEpochs"] = {
            str(batch): [target for target in TARGETS if target <= ceilings[str(batch)]]
            for batch in BATCHES
        }
        report["targetEpochs"] = sorted(
            {target for values in report["batchTargetEpochs"].values() for target in values}
        )
        report["updated"] = "2026-08-10"
        write_report(report_path, report)
        print(
            f"{model}: optimums={optimums} stopCeilings={ceilings} "
            f"source={source_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
