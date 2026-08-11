#!/usr/bin/env python3
"""Rewrite Pool-3B reports after historical effective-WD cleanup.

The canonical report rows correspond one-to-one with retained Weka output
directories.  Removed or failed attempts remain fully represented in attached
attempt history and in the top-level correction provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import date
from pathlib import Path
from typing import Any


REPORTS = {
    "153m": Path("reports/0802/data/wsd_batch_size_153m_pool3b.json"),
    "474m": Path("reports/0802/data/wsd_batch_size_474m_pool3b.json"),
    "1b": Path("reports/0802/data/wsd_batch_size_1b_pool3b.json"),
}
CANCELED_EXPERIMENTS = {
    "01KZQ0VJH41HFBX5M7C170A0FJ",
    "01KZPXECA5KCW58Z3ABN0F8426",
}


def replace_paths(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for source in sorted(mapping, key=len, reverse=True):
            if value == source or value.startswith(source + "/"):
                return mapping[source] + value[len(source) :]
        return value
    if isinstance(value, list):
        return [replace_paths(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, mapping) for key, item in value.items()}
    return value


def mark_canceled(sweep: dict[str, Any]) -> None:
    if sweep.get("beaker") not in CANCELED_EXPERIMENTS:
        return
    sweep["status"] = "canceled"
    sweep["failureClass"] = "user-canceled-pool3b-audit"
    sweep["failureReason"] = (
        "Canceled explicitly by the user after the Pool-3B effective weight-decay "
        "bug was confirmed; not an infrastructure or model failure."
    )
    sweep["reason"] = sweep["failureReason"]


def annotate_attempt(
    sweep: dict[str, Any], effective_wd: str, canonical_output: str | None
) -> dict[str, Any]:
    attempt = copy.deepcopy(sweep)
    mark_canceled(attempt)
    intended = str(attempt.get("wd"))
    attempt["historicalIntendedWd"] = intended
    attempt["effectiveWd"] = effective_wd
    attempt["outputRemovedFromWeka"] = True
    attempt["canonicalOutput"] = canonical_output
    for result in attempt.get("results", {}).values():
        if isinstance(result, dict):
            result["historicalIntendedWd"] = str(result.get("wd", intended))
            result["effectiveWd"] = effective_wd
    return attempt


def write_mirror(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA="
        + json.dumps(data, separators=(",", ":"), ensure_ascii=True)
        + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    groups_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in REPORTS}
    for group in audit["cleanupPlan"]["groups"]:
        groups_by_model[group["coordinate"]["model"]].append(group)

    deleted_outputs = {
        output for group in audit["cleanupPlan"]["groups"] for output in group["deleteOutputs"]
    }

    for model, report_path in REPORTS.items():
        report = json.loads(report_path.read_text())
        original_sweeps = report["batchSweeps"]
        by_output = {sweep.get("output"): sweep for sweep in original_sweeps}
        by_beaker = {sweep.get("beaker"): sweep for sweep in original_sweeps}
        path_mapping: dict[str, str] = {}
        canonical_specs: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        orphan_history: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for group in groups_by_model[model]:
            effective_wd = group["coordinate"]["effectiveWd"]
            corrected_output = group["correctedCanonicalOutput"]
            members = [
                by_output[output] for output in group["memberOutputs"] if output in by_output
            ]
            if corrected_output is not None:
                for output in group["memberOutputs"]:
                    path_mapping[output] = corrected_output
            canonical = by_beaker.get(group["canonicalBeaker"])
            if canonical is None:
                orphan_history.extend((member, group) for member in members)
                continue
            removed = [member for member in members if member is not canonical]
            canonical_specs.append((canonical, group, removed))

        canonical_rows: list[dict[str, Any]] = []
        archived_attempts: list[dict[str, Any]] = []
        for canonical_source, group, removed_sources in canonical_specs:
            effective_wd = group["coordinate"]["effectiveWd"]
            corrected_output = group["correctedCanonicalOutput"]
            historical_intended = str(canonical_source.get("wd"))
            canonical = replace_paths(copy.deepcopy(canonical_source), path_mapping)
            mark_canceled(canonical)
            canonical["historicalIntendedWd"] = historical_intended
            canonical["effectiveWd"] = effective_wd
            canonical["wd"] = effective_wd
            canonical["output"] = corrected_output
            canonical["consolidatedFromOutputs"] = group["consolidateFromOutputs"]
            canonical["checkpointDirectoryCanonicalized"] = True
            for result in canonical.get("results", {}).values():
                if isinstance(result, dict):
                    result["historicalIntendedWd"] = str(result.get("wd", historical_intended))
                    result["effectiveWd"] = effective_wd
                    result["wd"] = effective_wd

            history = list(canonical.get("attemptHistory", []))
            removed_attempts = [
                annotate_attempt(source, effective_wd, corrected_output)
                for source in removed_sources
            ]
            history.extend(removed_attempts)
            if history:
                canonical["attemptHistory"] = history
            archived_attempts.extend(removed_attempts)
            canonical_rows.append(canonical)

        # A coordinate with no valid retained output (currently failed BS512
        # E4 attempts) is attached to the closest preceding retained row for
        # the same batch so the HTML provenance table still renders it.
        for source, group in orphan_history:
            effective_wd = group["coordinate"]["effectiveWd"]
            attempt = annotate_attempt(source, effective_wd, None)
            archived_attempts.append(attempt)
            batch = int(group["coordinate"]["batchSequences"])
            epoch = int(group["coordinate"]["epoch"])
            candidates = [
                row
                for row in canonical_rows
                if int(row["batchSequences"]) == batch and int(row["activeEpoch"]) < epoch
            ]
            if candidates:
                owner = max(candidates, key=lambda row: int(row["activeEpoch"]))
                owner.setdefault("attemptHistory", []).append(attempt)

        canonical_rows.sort(
            key=lambda row: (
                int(row["batchSequences"]),
                int(row["activeEpoch"]),
                str(row.get("lr")),
                str(row.get("wd")),
            )
        )
        root_wd_by_batch: dict[str, str] = {}
        for row in canonical_rows:
            batch = str(row["batchSequences"])
            root_wd_by_batch.setdefault(batch, str(row["effectiveWd"]))

        historical_freeze = copy.deepcopy(report.get("wdFreezePolicy"))
        historical_initial = copy.deepcopy(
            report.get("poolPlan", {}).get("initialWeightDecayByBatch")
        )
        report["updated"] = date.today().isoformat()
        report["selection"] = (
            "Historical Pool-3B WD branches are collapsed by audited effective WD: "
            "optimizer-state restore made every continuation inherit the root 1B-pool "
            "checkpoint WD. These results therefore do not constitute a Pool-3B WD "
            "comparison. Future E1 work requires an exact same-model/batch/LR/WD 1B-pool "
            "source, and checkpoint loading fails on any model or optimizer hyperparameter "
            "mismatch."
        )
        report["historicalWdFreezePolicy"] = historical_freeze
        report["wdFreezePolicy"] = {
            "mode": "invalidated-by-effective-wd-audit",
            "effectiveWeightDecayByBatch": root_wd_by_batch,
            "futurePolicy": "exact-wd-matched-source-and-strict-config-assertion",
        }
        pool_plan = report.setdefault("poolPlan", {})
        pool_plan["historicalInitialWeightDecayByBatch"] = historical_initial
        pool_plan["initialWeightDecayByBatch"] = {
            batch: [wd] for batch, wd in root_wd_by_batch.items()
        }
        pool_plan["epochOneSourceWeightDecayByBatch"] = root_wd_by_batch
        report["batchSweeps"] = canonical_rows
        report["pool3bWeightDecayCorrection"] = {
            "status": "applied",
            "cause": (
                "Optimizer state loading overwrote the command-specified weight_decay; "
                "initial_lr alone was protected."
            ),
            "scope": "Pool-3B reports and output directories only; 1B-token-pool reports unchanged.",
            "registeredAttemptsBefore": len(original_sweeps),
            "canonicalCoordinatesAfter": len(canonical_rows),
            "removedOutputDirectories": sum(
                1 for attempt in archived_attempts if attempt.get("output") in deleted_outputs
            ),
            "effectiveWeightDecayByBatch": root_wd_by_batch,
            "removedRedundantSweeps": archived_attempts,
        }
        write_mirror(report_path, report)


if __name__ == "__main__":
    main()
