#!/usr/bin/env python3
"""Refresh nested-3B stopping evidence from the live trusted 1B reports."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "0802"
DATA = REPORT_DIR / "data"
TARGETS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88)
MODELS = {
    "153m": {
        "rank_mb": 16,
        "lr": {64: "2e-3", 128: "2e-3", 256: "2e-3", 512: "2e-3"},
        "initial_wd": {64: ["0.1", "0.033"], 128: ["0.1", "0.033"], 256: ["0.033", "0.01", "0.1"], 512: ["0.1", "0.033"]},
        "existing_wd": {64: "0.1", 128: "0.1", 256: "0.033", 512: "0.1"},
        "larger_minimum_epoch": {256: 12},
    },
    "474m": {
        "rank_mb": 16,
        "lr": {64: "2e-3", 128: "2e-3", 256: "2e-3", 512: "2e-3"},
        "initial_wd": {64: ["0.1", "0.033"], 128: ["0.1", "0.033", "0.3"], 256: ["0.1", "0.033", "0.333"], 512: ["0.3", "0.1"]},
        "existing_wd": {64: "0.1", 128: "0.1", 256: "0.1", 512: "0.3"},
        "larger_minimum_epoch": {128: 12, 256: 4},
    },
    "1b": {
        "rank_mb": 8,
        "lr": {64: "1e-3", 128: "1e-3", 256: "1e-3", 512: "5e-4"},
        "initial_wd": {64: ["0.3", "0.1"], 128: ["0.3", "0.1"], 256: ["0.333", "0.1"], 512: ["1.0", "0.333"]},
        "existing_wd": {64: "0.3", 128: "0.3", 256: "0.333", 512: "1.0"},
        "larger_minimum_epoch": {},
    },
}
BATCHES = (64, 128, 256, 512)


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


def repair_recovery_resume_checkpoints(report: dict) -> int:
    """Repair stale target paths inherited from pre-success recovery attempts."""
    repaired = 0
    for sweep in report.get("batchSweeps", []):
        if sweep.get("status") != "complete" or not sweep.get("recoveryOf"):
            continue
        retained = [int(step) for step in sweep.get("retainedPreDecaySteps", [])]
        if not retained:
            continue
        target_step = retained[-1]
        recovery_step = sweep.get("recoverySourceStep")
        if recovery_step is not None and int(recovery_step) >= target_step:
            continue
        checkpoint = f"{sweep['output']}/step{target_step}"
        if sweep.get("targetPreDecayCheckpoint") != checkpoint:
            sweep["targetPreDecayCheckpoint"] = checkpoint
            repaired += 1
        result = sweep.get("results", {}).get(str(sweep.get("activeEpoch")))
        if isinstance(result, dict) and result.get("resumeCheckpoint") != checkpoint:
            result["resumeCheckpoint"] = checkpoint
            repaired += 1
    return repaired


def refresh_html(model: str) -> None:
    path = REPORT_DIR / f"wsd_batch_size_{model}_pool3b.html"
    html = path.read_text()
    headers = "".join(f"<th>BS {batch}</th>" for batch in BATCHES)
    for first_header, body_id in (
        ("Epoch", "validation-summary"),
        ("Optimizer steps", "optimizer-step-summary"),
        ("Epoch", "coordinate-summary"),
    ):
        html = re.sub(
            rf'(<thead><tr><th>{re.escape(first_header)}</th>)'
            rf'(?:<th>BS \d+</th>)+'
            rf'(</tr></thead><tbody id="{body_id}">)',
            rf'\1{headers}\2',
            html,
        )
    path.write_text(html)


def main() -> None:
    for model, plan in MODELS.items():
        source_path = DATA / f"wsd_batch_size_{model}.json"
        report_path = DATA / f"wsd_batch_size_{model}_pool3b.json"
        source = json.loads(source_path.read_text())
        report = json.loads(report_path.read_text())
        repaired = repair_recovery_resume_checkpoints(report)
        optimums: dict[str, int] = {}
        ceilings: dict[str, int] = {}
        for batch in BATCHES:
            optimum, _ = trusted_optimum(source, batch, plan["lr"][batch])
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
        report["poolPlan"]["fixedLearningRateByBatch"] = {
            str(batch): plan["lr"][batch] for batch in BATCHES
        }
        report["poolPlan"]["initialWeightDecayByBatch"] = {
            str(batch): plan["initial_wd"][batch] for batch in BATCHES
        }
        report["poolPlan"]["epochOneSourceWeightDecayByBatch"] = {
            str(batch): plan["initial_wd"][batch] for batch in BATCHES
        }
        report["wdRepairPolicy"] = {
            "effectiveDate": "2026-08-10",
            "rule": (
                "Every missing WD starts at E1. A smaller WD is cut as soon as it "
                "fails to beat the existing WD at the same epoch. A larger maximum "
                "WD remains an independent exact-checkpoint trajectory through its "
                "minimum evidence epoch even when it loses at an earlier epoch."
            ),
            "byBatch": {
                str(batch): {
                    "existingWeightDecay": plan["existing_wd"][batch],
                    "neededWeightDecays": plan["initial_wd"][batch],
                    "smallerWeightDecays": [
                        wd
                        for wd in plan["initial_wd"][batch]
                        if Decimal(wd) < Decimal(plan["existing_wd"][batch])
                    ],
                    "largerWeightDecays": [
                        wd
                        for wd in plan["initial_wd"][batch]
                        if Decimal(wd) > Decimal(plan["existing_wd"][batch])
                    ],
                    "largerWeightDecayMinimumEpoch": plan[
                        "larger_minimum_epoch"
                    ].get(batch, 1),
                }
                for batch in BATCHES
            },
        }
        report["poolPlan"]["gpuTopologyByBatch"] = {
            str(batch): {
                "gpuCountPerNode": min(batch // plan["rank_mb"], 8),
                "nodeCount": 1,
                "totalGpuCount": min(batch // plan["rank_mb"], 8),
                "gradientAccumulation": batch
                // (min(batch // plan["rank_mb"], 8) * plan["rank_mb"]),
            }
            for batch in BATCHES
        }
        report["gpuTopology"] = {
            str(batch): {
                "gpuCountPerNode": min(batch // plan["rank_mb"], 8),
                "nodeCount": 1,
                "gpuCount": min(batch // plan["rank_mb"], 8),
                "gradientAccumulation": batch
                // (min(batch // plan["rank_mb"], 8) * plan["rank_mb"]),
            }
            for batch in BATCHES
        } | {"rankMicrobatchSequences": plan["rank_mb"]}
        report["batchTargetEpochs"] = {
            str(batch): [target for target in TARGETS if target <= ceilings[str(batch)]]
            for batch in BATCHES
        }
        report["targetEpochs"] = sorted(
            {target for values in report["batchTargetEpochs"].values() for target in values}
        )
        report["summaryBatches"] = list(BATCHES)
        report["setup"] = report["setup"].replace(
            "64, 128, and 256", "64, 128, 256, and 512"
        )
        report["updated"] = "2026-08-10"
        write_report(report_path, report)
        refresh_html(model)
        print(
            f"{model}: optimums={optimums} stopCeilings={ceilings} "
            f"repairedResumeFields={repaired} source={source_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
