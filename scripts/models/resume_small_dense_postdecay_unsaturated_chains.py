#!/usr/bin/env python3
"""Resume completed locked-WD chains whose POST series was still improving."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
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


def postdecay_is_improving(selection: dict[str, Any]) -> bool:
    sources = [int(epoch) for epoch in selection.get("postDecaySourceEpochs", [])]
    values = selection.get("postDecayValidationExact") or {}
    if len(sources) != 3 or any(str(epoch) not in values for epoch in sources):
        raise RuntimeError("post-decay continuation requires exactly three results")
    return Decimal(str(values[str(sources[-1])])) < Decimal(
        str(values[str(sources[-2])])
    )


def command_for(model: str, record: dict[str, Any], revision: str) -> list[str]:
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
        "--postdecay-policy-resume",
        "--locked-wd",
        str(record["lockedWd"]),
        "--historical-predecay-through-epoch",
        str(record["historicalPreDecayThroughEpoch"]),
        "--name",
        f"dense-{model}-bs{batch}-postdecay-resume-{stamp}",
    ]


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise SystemExit("--revision must be the full 40-character pushed revision")
    summaries: list[str] = []
    for model, path in REPORTS.items():
        report = json.loads(path.read_text())
        for record in report.get("adaptiveDrWtEmbedWdChains", []):
            if record.get("policy") != POLICY:
                continue
            chain = f"{model} BS{record['batchSequences']}"
            if not record.get("needsPolicyResume"):
                summaries.append(f"{chain}: no post-decay continuation required")
                continue
            if record.get("beakerStatus") != "complete":
                summaries.append(f"{chain}: waiting for completed Beaker experiment")
                continue
            selection = record.get("supersededPostDecaySelection") or {}
            if not postdecay_is_improving(selection):
                raise RuntimeError(f"{chain}: resume flag contradicts post-decay results")
            command = command_for(model, record, args.revision)
            if args.submit_if_ready:
                subprocess.run(command, check=True)
                summaries.append(
                    f"{chain}: resumed after provisional E{selection['saturationEpoch']} stop"
                )
            else:
                summaries.append(f"{chain}: READY " + " ".join(command))
    print("\n".join(summaries) if summaries else "no locked-WD chains registered")


if __name__ == "__main__":
    main()
