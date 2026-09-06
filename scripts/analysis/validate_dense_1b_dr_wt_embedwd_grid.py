#!/usr/bin/env python3
"""Validate the guarded Dense-1B all-POST plan and report provenance."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

REPORT = Path("reports/0802/data/wsd_data_loader_1b.json")
HTML = Path("reports/0802/wsd_data_loader_1b.html")
CHARTS = Path("reports/0802/data_loader_charts.js")
POLICY = "dense_1b_all_postdecay_saturation_v1"
MANIFESTS = {
    128: Path("scripts/models/manifests/dense-1b-bs128-dr-wt-embwd-grid.json"),
    256: Path("scripts/models/manifests/dense-1b-bs256-dr-wt-embwd-grid.json"),
}
CONDITIONAL_MANIFESTS = (
    Path("scripts/models/manifests/dense-1b-bs512-original-lr2e-3-wd0.333-pdpost.json"),
    Path("scripts/models/manifests/dense-1b-bs512-dr-wt-embwd-lr2e-3-grid.json"),
)
EXPECTED = {
    128: {("1e-3", "0.3"), ("1e-3", "1.0")},
    256: {
        ("1e-3", "0.3"),
        ("1e-3", "1.0"),
        ("2e-3", "0.3"),
        ("2e-3", "1.0"),
    },
}
EXPECTED_CONDITIONAL = {
    ("Original", "2e-3", "0.333"),
    ("DR+WT+EmbedWD", "2e-3", "0.333"),
    ("DR+WT+EmbedWD", "2e-3", "1.0"),
}
EXACT_ORIGINAL_E1 = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_e1_lr2e-3_wd0.333_"
    "warmup48_e1_e2_lrup_wd0333_r25/step428"
)
EXACT_ORIGINAL_E2 = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_e2_lr2e-3_wd0.333_"
    "warmup48_e1_e2_lrup_wd0333_r25/step858"
)
EXACT_ORIGINAL_E4 = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_e4_lr2e-3_wd0.333_"
    "warmup48_e4_lrup_infra_r27/step1716"
)
EXACT_ORIGINAL_E8 = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_e8_lr2e-3_wd0.333_"
    "warmup48_e8_lr2_wd0333_r30/step3432"
)


def validate_manifest(path: Path, expected: set[tuple[str, str]]) -> dict:
    manifest = json.loads(path.read_text())
    assert manifest["policy"] == POLICY
    assert manifest["comparisonPolicy"] == "post_decay_only"
    assert manifest["checkpointOnlyEpochs"] == [1, 2, 4]
    assert manifest["postDecayStartEpoch"] == 8
    assert manifest["postDecayEvaluation"] == "every_scheduled_frontier_from_e8"
    assert manifest["postDecaySourceCount"] == 3
    assert manifest["postDecaySaturationCriterion"] == "strict_non_improvement"
    assert manifest["nprocPerNode"] == 8
    assert manifest["rankMicrobatchSequences"] == 8
    assert manifest["gradientAccumulation"] == int(manifest["globalSequences"]) // 64
    assert manifest["maxEpoch"] == 256
    assert {(str(item["lr"]), str(item["wd"])) for item in manifest["coordinates"]} == expected
    assert all(item.get("output") for item in manifest["coordinates"])
    assert all(Decimal(str(item["wd"])) <= Decimal("1.0") for item in manifest["coordinates"])
    return manifest


def main() -> None:
    report = json.loads(REPORT.read_text())
    assert report["dense1bPdPostPolicy"]["policy"] == POLICY
    assert report["dense1bPdPostPolicy"]["parallelPrimaryCoordinates"] == 6
    assert report["dense1bPdPostPolicy"]["checkpointOnlyEpochs"] == [1, 2, 4]
    assert report["dense1bPdPostPolicy"]["postDecayStartEpoch"] == 8
    assert report["dense1bPdPostPolicy"]["saturationDecisionGroup"] == "post_decay_only"
    columns = [column["key"] for column in report["columns"]]
    for batch in EXPECTED:
        assert columns.index(f"drwtembwd{batch}") == columns.index(f"dr{batch}") + 1
    chains = {record["id"]: record for record in report.get("drWtEmbedWdGridChains", [])}
    assert {
        "dense-1b-bs128-dr-wt-embwd-grid",
        "dense-1b-bs256-dr-wt-embwd-grid",
        "dense-1b-bs512-lr2e-3-conditional-followup",
    }.issubset(chains)
    for batch, expected in EXPECTED.items():
        chain = chains[f"dense-1b-bs{batch}-dr-wt-embwd-grid"]
        assert chain["policy"] == POLICY
        assert chain["triggerThreshold"] == 10
        assert chain["gpuCountPerCoordinate"] == 8
        assert chain["nodeCountPerCoordinate"] == 1
        assert chain["rankMicrobatchSequences"] == 8
        assert chain["gradientAccumulation"] == batch // 64
        assert chain["parallelCoordinates"] == len(expected)
        assert chain["comparisonPolicy"] == "post_decay_only"
        assert chain["checkpointOnlyEpochs"] == [1, 2, 4]
        assert chain["postDecayStartEpoch"] == 8
        assert chain["postDecayEvaluation"] == "every_scheduled_frontier_from_e8"
        assert chain["postDecaySourceCount"] == 3
        assert chain["maxEpoch"] == 256
        assert {(item["lr"], item["wd"]) for item in chain["coordinates"]} == expected
    conditional = chains["dense-1b-bs512-lr2e-3-conditional-followup"]
    assert conditional["trigger"] == "terminal_bs256_selected_lr_2e-3"
    assert {
        (item["variant"], item["lr"], item["wd"]) for item in conditional["coordinates"]
    } == EXPECTED_CONDITIONAL
    primary_runs = [
        run
        for run in report["runs"]
        if run.get("policy") == POLICY and int(run["batchSequences"]) in {128, 256}
    ]
    assert len(primary_runs) == 6
    assert {
        (int(run["batchSequences"]), str(run["lr"]), str(run["wd"])) for run in primary_runs
    } == {(batch, lr, wd) for batch, values in EXPECTED.items() for lr, wd in values}
    for run in primary_runs:
        assert run["gpuCount"] == 8 and run["nodeCount"] == 1
        assert run["comparisonPolicy"] == "post_decay_only"
        assert run["checkpointOnlyEpochs"] == [1, 2, 4]
        assert run["postDecayStartEpoch"] == 8
        assert run["postDecayEvaluation"] == "every_scheduled_frontier_from_e8"
        assert run["postDecaySourceCount"] == 3
        assert Decimal(str(run["wd"])) <= Decimal("1.0")
        assert all(int(epoch) >= 8 for epoch in run.get("postDecayResults", {}))
    conditional_runs = [
        run
        for run in report["runs"]
        if run.get("policy") == POLICY and int(run["batchSequences"]) == 512
    ]
    assert len(conditional_runs) == 3
    assert {
        (run["variant"], str(run["lr"]), str(run["wd"])) for run in conditional_runs
    } == EXPECTED_CONDITIONAL
    for run in conditional_runs:
        assert run["comparisonPolicy"] == "post_decay_only"
        assert run["checkpointOnlyEpochs"] == [1, 2, 4]
        assert run["postDecayStartEpoch"] == 8
        assert run["postDecayEvaluation"] == "every_scheduled_frontier_from_e8"
        assert run["postDecaySourceCount"] == 3
        assert Decimal(str(run["wd"])) <= Decimal("1.0")
        assert all(int(epoch) >= 8 for epoch in run.get("postDecayResults", {}))
    original = next(run for run in conditional_runs if run["variant"] == "Original")
    assert original["sourceCheckpoint"] == EXACT_ORIGINAL_E8
    assert original["dynamicRepacking"] is False
    assert original["weightTying"] is False
    assert original["decayEmbeddings"] is False
    for batch, path in MANIFESTS.items():
        manifest = validate_manifest(path, EXPECTED[batch])
        assert int(manifest["globalSequences"]) == batch
        assert manifest["variant"] == "DR+WT+EmbedWD"
    original_manifest = validate_manifest(CONDITIONAL_MANIFESTS[0], {("2e-3", "0.333")})
    assert original_manifest["variant"] == "Original"
    historical = original_manifest["coordinates"][0]["historicalPreDecay"]
    assert historical == [
        {
            "epoch": 1,
            "checkpoint": EXACT_ORIGINAL_E1,
            "experiment": "01KZGE4PS4WSJG3HHHEDXGTXY7",
            "checkpointOnly": True,
        },
        {
            "epoch": 2,
            "checkpoint": EXACT_ORIGINAL_E2,
            "experiment": "01KZGE4PS4WSJG3HHHEDXGTXY7",
            "checkpointOnly": True,
        },
        {
            "epoch": 4,
            "checkpoint": EXACT_ORIGINAL_E4,
            "experiment": "01KZGN0KAZKN71JJ5X32K5EQGG",
            "checkpointOnly": True,
        },
        {
            "epoch": 8,
            "checkpoint": EXACT_ORIGINAL_E8,
            "experiment": "01KZGV0B3Q7ESNHR83KVQJRRE8",
        },
    ]
    drwt_manifest = validate_manifest(
        CONDITIONAL_MANIFESTS[1], {("2e-3", "0.333"), ("2e-3", "1.0")}
    )
    assert drwt_manifest["variant"] == "DR+WT+EmbedWD"
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
    assert "wsd_data_loader_1b.js?v=20260906-redecay-results" in html
    assert "data_loader_charts.js?v=20260826-coordinate-grid-status" in html
    assert "data_loader_early_frontier.js?v=20260906-v1" in html
    assert "Val CE (POST | PD)" in html
    charts = CHARTS.read_text()
    assert "Unknown" in charts
    assert "[POST] |" in charts and "[PD]" in charts
    assert "result.postDecay" in charts and "result.preDecay" in charts
    print("Dense-1B guarded all-POST-from-E8 plan validated")


if __name__ == "__main__":
    main()
