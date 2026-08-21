#!/usr/bin/env python3
"""Record a non-destructive transition request for all eight adaptive chains."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

REPORTS = {
    "474m": Path("reports/0802/data/wsd_batch_size_474m.json"),
    "153m": Path("reports/0802/data/wsd_batch_size_153m.json"),
}
POLICY = "locked_wd_predecay_saturation_v1"


def latest_selected(record: dict[str, Any]) -> tuple[int, str]:
    frontiers = record.get("frontiers", {})
    if not frontiers:
        raise RuntimeError(f"{record['id']} has no resolved WD frontier")
    epoch = max(int(value) for value in frontiers)
    return epoch, str(frontiers[str(epoch)]["selectedWd"])


def update_record(record: dict[str, Any], requested_at: str) -> str:
    if record.get("policy") == POLICY:
        return f"{record['id']}: already transitioned"
    frontier_epoch, locked_wd = latest_selected(record)
    if Decimal(locked_wd) > Decimal("1.0"):
        raise RuntimeError(f"{record['id']} selected forbidden WD{locked_wd}")
    active_epoch = record.get("activeEpoch")
    active_wd = record.get("activeWd")
    if active_epoch is None or active_wd is None:
        raise RuntimeError(f"{record['id']} has no active stage to finish safely")
    existing = dict(record.get("policyTransition", {}))
    if existing.get("status") == "awaiting-current-stage":
        return (
            f"{record['id']}: already awaiting E{existing['awaitStage']['epoch']} "
            f"WD{existing['awaitStage']['wd']}"
        )
    record["policyTransition"] = {
        "requestedAt": requested_at,
        "status": "awaiting-current-stage",
        "oldExperiment": record["experiment"],
        "lockedWd": locked_wd,
        "lastResolvedFrontierEpoch": frontier_epoch,
        "awaitStage": {
            "epoch": int(active_epoch),
            "wd": str(active_wd),
            "progressAtRequest": dict(record.get("progress", {})),
        },
        "handoffRule": (
            "Allow the active candidate to finish, then stop the old adaptive job. "
            "Any automatically started successor is considered newly started and may "
            "be canceled during the handoff."
        ),
    }
    record["pendingPolicy"] = POLICY
    record["pendingLockedWd"] = locked_wd
    return f"{record['id']}: await E{active_epoch} WD{active_wd}; then lock WD{locked_wd}"


def main() -> None:
    requested_at = datetime.now(tz=UTC).isoformat()
    summaries: list[str] = []
    for path in REPORTS.values():
        report = json.loads(path.read_text())
        for record in report.get("adaptiveDrWtEmbedWdChains", []):
            summaries.append(update_record(record, requested_at))
        report["updated"] = datetime.now(tz=UTC).date().isoformat()
        path.write_text(json.dumps(report, indent=2) + "\n")
        path.with_suffix(".js").write_text(
            "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
        )
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
