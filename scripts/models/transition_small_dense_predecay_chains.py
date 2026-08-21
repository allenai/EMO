#!/usr/bin/env python3
"""Replace adaptive chains only after their transition-request stage completes."""

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
POLICY = "locked_wd_predecay_saturation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--submit-if-ready", action="store_true")
    return parser.parse_args()


def stage_complete(record: dict[str, Any], transition: dict[str, Any]) -> bool:
    awaited = transition["awaitStage"]
    by_epoch = record.get("results", {}).get(str(awaited["epoch"]), {})
    result = by_epoch.get(str(awaited["wd"]), {})
    return result.get("status") == "complete"


def command_for(model: str, record: dict[str, Any], revision: str) -> list[str]:
    transition = record["policyTransition"]
    batch = int(record["batchSequences"])
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return [
        ".venv/bin/python",
        "scripts/models/submit_small_dense_dr_wt_embedwd_chain.py",
        "--model",
        model,
        "--global-sequences",
        str(batch),
        "--revision",
        revision,
        "--resume-experiment",
        str(record["experiment"]),
        "--stop-existing",
        "--register",
        "--predecay-policy-replacement",
        "--locked-wd",
        str(transition["lockedWd"]),
        "--historical-predecay-through-epoch",
        str(transition["awaitStage"]["epoch"]),
        "--name",
        f"dense-{model}-bs{batch}-lockedwd-predecay-{stamp}",
    ]


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise SystemExit("--revision must be the full 40-character pushed revision")
    summaries: list[str] = []
    for model, path in REPORTS.items():
        report = json.loads(path.read_text())
        for record in report.get("adaptiveDrWtEmbedWdChains", []):
            if record.get("policy") == POLICY:
                summaries.append(f"{record['id']}: transitioned {record['experiment']}")
                continue
            transition = record.get("policyTransition") or {}
            if transition.get("status") != "awaiting-current-stage":
                summaries.append(f"{record['id']}: no pending transition")
                continue
            awaited = transition["awaitStage"]
            if not stage_complete(record, transition):
                summaries.append(
                    f"{record['id']}: waiting for E{awaited['epoch']} WD{awaited['wd']}"
                )
                continue
            command = command_for(model, record, args.revision)
            if args.submit_if_ready:
                subprocess.run(command, check=True)
                summaries.append(
                    f"{record['id']}: replaced after E{awaited['epoch']} WD{awaited['wd']}"
                )
            else:
                summaries.append(f"{record['id']}: READY " + " ".join(command))
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
