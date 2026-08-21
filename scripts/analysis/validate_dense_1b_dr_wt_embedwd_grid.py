#!/usr/bin/env python3
"""Validate the deferred Dense-1B BS128/256 DR+WT+EmbedWD grid and report."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

REPORT = Path("reports/0802/data/wsd_data_loader_1b.json")
HTML = Path("reports/0802/wsd_data_loader_1b.html")
MANIFESTS = {
    128: Path("scripts/models/manifests/dense-1b-bs128-dr-wt-embwd-grid.json"),
    256: Path("scripts/models/manifests/dense-1b-bs256-dr-wt-embwd-grid.json"),
}
EXPECTED = {
    128: {("1e-3", "0.3"), ("1e-3", "1.0")},
    256: {
        ("1e-3", "0.333"),
        ("1e-3", "1.0"),
        ("2e-3", "0.333"),
        ("2e-3", "1.0"),
    },
}
POLICY_HOLD = "held_by_locked_wd_predecay_policy_2026_08_21"


def main() -> None:
    report = json.loads(REPORT.read_text())
    columns = [column["key"] for column in report["columns"]]
    for batch in EXPECTED:
        assert columns.index(f"drwtembwd{batch}") == columns.index(f"dr{batch}") + 1
    chains = report.get("drWtEmbedWdGridChains", [])
    assert len(chains) == 2
    for chain in chains:
        batch = int(chain["batchSequences"])
        assert chain["triggerThreshold"] == 5
        assert chain["gpuCount"] == 8
        assert chain["rankMicrobatchSequences"] == 8
        assert chain["gradientAccumulation"] == batch // 64
        assert chain["weightTying"] is True
        assert chain["decayEmbeddings"] is True
        assert {(item["lr"], item["wd"]) for item in chain["coordinates"]} == EXPECTED[batch]
        if not chain.get("experiment"):
            assert chain.get("status") == "held"
            assert chain.get("policyHold") == POLICY_HOLD
    registered = {
        (int(run["batchSequences"]), str(run["lr"]), str(run["wd"]))
        for run in report["runs"]
        if run.get("method") in {"drwtembwd128", "drwtembwd256"}
    }
    expected_registered = {
        (batch, lr, wd) for batch, values in EXPECTED.items() for lr, wd in values
    }
    assert registered == expected_registered
    assert all(Decimal(wd) <= Decimal("1.0") for _, _, wd in registered)
    for batch, path in MANIFESTS.items():
        manifest = json.loads(path.read_text())
        assert int(manifest["globalSequences"]) == batch
        assert {(item["lr"], item["wd"]) for item in manifest["coordinates"]} == EXPECTED[batch]
        assert manifest["nprocPerNode"] == 8
        assert manifest["rankMicrobatchSequences"] == 8
    html = HTML.read_text()
    grouped_headers = [
        row
        for row in re.findall(r"<thead><tr>(.*?)</tr></thead>", html)
        if "BS64 · Original" in row
    ]
    assert grouped_headers
    assert all(row.count("<th>") == 19 for row in grouped_headers)
    assert all("BS128 · DR+WT+EmbedWD" in row for row in grouped_headers)
    assert all("BS256 · DR+WT+EmbedWD" in row for row in grouped_headers)
    assert ":nth-child(6),:nth-child(9),:nth-child(12),:nth-child(17)" in html
    print("Dense-1B deferred DR+WT+EmbedWD grids validated")


if __name__ == "__main__":
    main()
