#!/usr/bin/env python3
"""Validate the two small-dense Step 1-1 report registries and mirrors."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

REPORTS = (
    Path("reports/0802/data/wsd_batch_size_153m.json"),
    Path("reports/0802/data/wsd_batch_size_474m.json"),
)
EXPECTED_BATCHES = [32, 64, 128, 256, 512]
SUMMARY_BATCHES = EXPECTED_BATCHES
POST_E16_TARGETS = [24, 32, 40, 48, 56, 64]
BASE_DISPLAY_EPOCHS = [
    0.125,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
]
BS32_DISPLAY_EPOCHS = [epoch for epoch in BASE_DISPLAY_EPOCHS if epoch >= 1]
AVG8_TASKS = (
    "arc_challenge",
    "arc_easy",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
ADAPTIVE_SEARCH = "small-model-adaptive-coordinate"
FRACTIONAL_SEARCH = "small-model-selected-e1-fractional-chain"
LOCKED_POLICIES = {
    "locked_wd_predecay_saturation_v1",
    "locked_wd_requested_postdecay_finalizer_v1",
    "locked_wd_all_postdecay_saturation_v1",
}
REPORT_RENDERERS = (
    Path("reports/0802/batch_size_charts.js"),
    Path("reports/0802/small_dense_dr_wt_embedwd_charts.js"),
)


def numeric(value: object) -> Decimal:
    return Decimal(str(value))


for renderer_path in REPORT_RENDERERS:
    renderer = renderer_path.read_text()
    for policy in LOCKED_POLICIES:
        assert policy in renderer, (
            f"{renderer_path} does not render locked-WD policy {policy}; "
            "this would hide retained [PD]/[POST] report results"
        )


for path in REPORTS:
    report = json.loads(path.read_text())
    base_display_epochs = BASE_DISPLAY_EPOCHS + (
        [72, 80, 88, 96, 104, 112, 120, 128] if "153m" in path.stem else []
    )
    observed_policy_epochs: dict[int, set[int]] = {}
    for chain in report.get("adaptiveDrWtEmbedWdChains", []):
        if chain.get("policy") not in LOCKED_POLICIES:
            continue
        batch = int(chain["batchSequences"])
        observed = observed_policy_epochs.setdefault(batch, set())
        for key in ("preDecayResults", "postDecayResults"):
            observed.update(int(epoch) for epoch in chain.get(key, {}))
    display_epochs = sorted(
        {
            *base_display_epochs,
            *(epoch for values in observed_policy_epochs.values() for epoch in values),
        },
        key=float,
    )
    mirror_text = path.with_suffix(".js").read_text()
    prefix = "window.ICSL_REPORT_DATA="
    assert mirror_text.startswith(prefix) and mirror_text.endswith(";\n"), path
    mirror = json.loads(mirror_text[len(prefix) : -2])
    assert mirror == report, f"JSON/JS mirror mismatch: {path}"
    assert report.get("selectionPolicy", {}).get("nondecreasingWd") is True, path
    assert numeric(report.get("selectionPolicy", {}).get("maxLearningRate")) == Decimal(
        "0.002"
    ), path
    policy = report["selectionPolicy"]
    assert set(map(int, policy.get("learningRateFreezeAfterEpoch", {}))) == {
        32,
        64,
        128,
        256,
        512,
    }, path
    assert set(map(int, policy.get("maxWeightDecayByBatch", {}))) == {
        32,
        64,
        128,
        256,
        512,
    }, path
    assert policy.get("postE16TargetEpochs") == POST_E16_TARGETS, path
    if "153m" in path.stem:
        assert policy.get("learningRateFreezeAfterEpoch") == {
            "32": 1,
            "64": 8,
            "128": 12,
            "256": 12,
            "512": 2,
        }
        assert policy.get("singleCoordinateAfterEpoch") == {
            "32": 1,
            "64": 16,
            "128": 16,
            "256": 16,
            "512": 16,
        }
        assert policy.get("holdAfterEpoch") == {
            "128": 80,
            "512": 48,
        }, policy.get("holdAfterEpoch")
        fixed = policy.get("fixedContinuationChains", [])
        assert fixed == [
            {
                "batchSequences": 64,
                "lr": "2e-3",
                "wd": "0.1",
                "afterEpoch": 64,
                "throughEpoch": 128,
                "stopOnNonImprovement": True,
                "tune": False,
            },
            {
                "batchSequences": 128,
                "lr": "2e-3",
                "wd": "0.1",
                "afterEpoch": 64,
                "throughEpoch": 80,
                "stopOnNonImprovement": True,
                "tune": False,
            },
            {
                "batchSequences": 128,
                "lr": "2e-3",
                "wd": "0.3",
                "afterEpoch": 16,
                "throughEpoch": 80,
                "stopOnNonImprovement": True,
                "tune": False,
            },
            {
                "batchSequences": 256,
                "lr": "2e-3",
                "wd": "0.3",
                "afterEpoch": 1,
                "throughEpoch": 16,
                "stopOnNonImprovement": False,
                "tune": False,
            },
            {
                "batchSequences": 256,
                "lr": "2e-3",
                "wd": "0.033",
                "afterEpoch": 16,
                "throughEpoch": 32,
                "stopOnNonImprovement": True,
                "tune": False,
            }
        ], fixed
    else:
        assert policy.get("learningRateFreezeAfterEpoch") == {
            "32": 1,
            "64": 2,
            "128": 4,
            "256": 4,
            "512": 4,
        }
        assert policy.get("singleCoordinateAfterEpoch") == {
            "32": 1,
            "64": 16,
            "128": 16,
            "256": 16,
            "512": 16,
        }
        assert not policy.get("holdAfterEpoch"), policy.get("holdAfterEpoch")
        fixed = policy.get("fixedContinuationChains", [])
        assert fixed == [
            {
                "batchSequences": 512,
                "lr": "2e-3",
                "wd": "1.0",
                "afterEpoch": 1,
                "throughEpoch": 16,
                "stopOnNonImprovement": False,
                "tune": False,
            }
        ], fixed
    assert report.get("targetEpochs") == display_epochs, path
    assert report.get("summaryBatches") == SUMMARY_BATCHES, path
    assert set(map(int, report.get("batchTargetEpochs", {}))) == set(EXPECTED_BATCHES)
    expected_batch_epochs = {}
    for batch in EXPECTED_BATCHES:
        baseline_epochs = (
            base_display_epochs
            if batch == 32 and "474m" in path.stem
            else BS32_DISPLAY_EPOCHS
            if batch == 32
            else base_display_epochs
            if "474m" in path.stem
            else BASE_DISPLAY_EPOCHS + [72, 80]
            if "153m" in path.stem and batch == 128
            else base_display_epochs
            if "153m" in path.stem and batch == 64
            else [epoch for epoch in BASE_DISPLAY_EPOCHS if epoch <= 48]
            if "153m" in path.stem and batch == 512
            else BASE_DISPLAY_EPOCHS
        )
        expected_batch_epochs[str(batch)] = sorted(
            {*baseline_epochs, *observed_policy_epochs.get(batch, set())}, key=float
        )
    assert report.get("batchTargetEpochs") == expected_batch_epochs, path

    active_keys: set[tuple[int, Decimal, Decimal, Decimal]] = set()
    fractional_keys: set[tuple[int, Decimal, Decimal, Decimal]] = set()
    for sweep in report.get("batchSweeps", []):
        if sweep.get("search") == ADAPTIVE_SEARCH:
            key = (
                int(sweep["batchSequences"]),
                numeric(sweep["lr"]),
                numeric(sweep["wd"]),
                numeric(sweep["activeEpoch"]),
            )
            assert key not in active_keys, f"duplicate small-model active tuple {key}"
            active_keys.add(key)
        elif sweep.get("search") == FRACTIONAL_SEARCH:
            batch = int(sweep["batchSequences"])
            expected_targets = (
                [Decimal("0.125"), Decimal("0.25"), Decimal("0.5")]
                if batch in {32, 64, 128}
                else [Decimal("0.25"), Decimal("0.5")]
            )
            assert batch in {32, 64, 128, 256}, sweep
            targets = [numeric(epoch) for epoch in sweep.get("fractionalTargets", [])]
            assert targets == expected_targets, (path, batch, targets)
            endpoints = sweep.get("fractionalEndpoints", {})
            assert set(map(numeric, endpoints)) == set(targets), (path, batch, endpoints)
            selected = sweep.get("selectedEpochOne", {})
            assert numeric(selected.get("lr")) == numeric(sweep["lr"]), sweep
            assert numeric(selected.get("wd")) == numeric(sweep["wd"]), sweep
            for target in targets:
                key = (batch, numeric(sweep["lr"]), numeric(sweep["wd"]), target)
                assert key not in fractional_keys, f"duplicate fractional tuple {key}"
                fractional_keys.add(key)
        attempt_history = sweep.get("attemptHistory", [])
        if attempt_history:
            assert sweep.get("recoveryOf") == attempt_history[-1].get("beaker"), sweep
            for attempt in attempt_history:
                assert attempt.get("status") in {"failed", "canceled"}, attempt
                assert str(attempt.get("failureClass", "")).startswith(
                    ("infrastructure-", "preflight-", "user-policy-pause")
                ), attempt
                assert attempt.get("beaker"), attempt
        for result in sweep.get("results", {}).values():
            downstream_bpb = result.get("downstreamBpb")
            if not isinstance(downstream_bpb, dict) or result.get("avg8Bpb") is None:
                continue
            expected = sum(float(downstream_bpb[task]) for task in AVG8_TASKS) / 8
            assert abs(expected - float(result["avg8Bpb"])) < 1e-9, (
                path,
                sweep.get("batchSequences"),
                sweep.get("lr"),
                sweep.get("wd"),
                result.get("epoch"),
            )

    # The reports live one directory above data/ and use the shared renderer.
    html = path.parent.parent / path.name.replace(".json", ".html")
    html_text = html.read_text()
    assert f'data/{path.with_suffix(".js").name}' in html_text, html
    assert 'src="batch_size_charts.js' in html_text, html
    assert "Downstream metrics by epoch and batch size" in html_text, html
    for downstream_id in (
        "epoch-hs-accuracy-summary",
        "epoch-hs-bpb-summary",
        "epoch-avg8-accuracy-summary",
        "epoch-avg8-bpb-summary",
    ):
        assert f'id="{downstream_id}"' in html_text, (html, downstream_id)
    assert "Chosen (LR, WD) by epoch and batch size" in html_text, html
    assert 'id="coordinate-summary"' in html_text, html
    adaptive_batches = {
        int(chain["batchSequences"])
        for chain in report.get("adaptiveDrWtEmbedWdChains", [])
    }
    for batch in SUMMARY_BATCHES:
        assert html_text.count(f"<th>BS {batch} · Original</th>") >= 7, (html, batch)
        if batch in adaptive_batches:
            assert (
                html_text.count(f"<th>BS {batch} · DR+WT+EmbedWD</th>") >= 7
            ), (html, batch)

print("small-dense Step 1-1 reports validated")
