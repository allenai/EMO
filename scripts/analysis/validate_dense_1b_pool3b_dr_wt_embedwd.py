#!/usr/bin/env python3
"""Validate the isolated Dense-1B Pool-3B plan and report mirror."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "models"))
import run_dense_1b_pool3b_dr_wt_embedwd as runner

REPORT = ROOT / "reports/0802/data/wsd_data_loader_1b_pool3b_drwtembwd.json"
MIRROR = REPORT.with_suffix(".js")
HTML = ROOT / "reports/0802/wsd_data_loader_1b_pool3b_drwtembwd.html"
MANIFESTS = (
    ROOT / "scripts/models/manifests/dense-1b-pool3b-bs64-dr-wt-embwd.json",
    ROOT / "scripts/models/manifests/dense-1b-pool3b-bs128-dr-wt-embwd.json",
)


def main() -> None:
    configs = [json.loads(path.read_text()) for path in MANIFESTS]
    for config in configs:
        runner.validate_config(config)
    report = json.loads(REPORT.read_text())
    assert report["policy"] == runner.POLICY
    assert report["poolPlan"]["uniqueTokens"] == runner.POOL_TOKENS
    assert report["poolPlan"]["decayCap"] is None
    assert report["poolPlan"]["requestedCheckpointEpochs"] == [4, 8, 12, 16, 20, 24, 28, 32]
    assert report["poolPlan"]["postDecayEpochs"][:3] == [32, 48, 64]
    runs = report["runs"]
    assert len(runs) == 2
    assert {int(run["batchSequences"]) for run in runs} == {64, 128}
    for run in runs:
        assert Decimal(str(run["lr"])) == Decimal("1e-3")
        assert Decimal(str(run["wd"])) == Decimal("0.3")
        assert run["weightTying"] and run["decayEmbeddings"] and run["dynamicRepacking"]
        assert run["comparisonPolicy"] == "post_decay_only"
        assert int(run["postDecayStartEpoch"]) == 32
        for epoch, result in run.get("postDecayResults", {}).items():
            assert int(epoch) >= 32
            assert result.get("comparisonGroup") == "post_decay"
        for epoch, result in run.get("preDecayResults", {}).items():
            assert result.get("comparisonGroup") == "checkpoint_provenance_only"
            combined = run.get("results", {}).get(str(epoch), {})
            assert "validationExact" not in combined, "PD leaked into POST summary"
    prefix = "window.ICSL_POOL3B_DATA="
    mirror = MIRROR.read_text().strip()
    assert mirror.startswith(prefix) and mirror.endswith(";")
    assert json.loads(mirror[len(prefix) : -1]) == report
    html = HTML.read_text()
    for asset in (
        "data/wsd_data_loader_1b_pool3b_drwtembwd.js",
        "data_loader_pool3b_drwtembwd_charts.js",
    ):
        assert asset in html
    assert not re.search(r"wsd_data_loader_1b\.js", html), "new report must stay isolated"
    print("validated Dense-1B Pool-3B DR+WT+EmbedWD plan and report")


if __name__ == "__main__":
    main()
