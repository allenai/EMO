#!/usr/bin/env python3
"""Validate the registered small-dense DR+WT+EmbedWD adaptive chains."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

MODELS = ("474m", "153m")
BATCHES = (64, 128, 256, 512)
ROOT = Path(__file__).resolve().parents[2]
POLICY = "locked_wd_predecay_saturation_v1"
FINALIZER_POLICY = "locked_wd_requested_postdecay_finalizer_v1"
POLICIES = {POLICY, FINALIZER_POLICY}


def validate_model(model: str) -> None:
    path = ROOT / "reports" / "0802" / "data" / f"wsd_batch_size_{model}.json"
    report = json.loads(path.read_text())
    records = report.get("adaptiveDrWtEmbedWdChains", [])
    if len(records) != 4:
        raise ValueError(f"{model}: expected four adaptive chains, found {len(records)}")
    by_batch = {int(record["batchSequences"]): record for record in records}
    if sorted(by_batch) != list(BATCHES):
        raise ValueError(f"{model}: adaptive chain batches are {sorted(by_batch)}")
    experiments: set[str] = set()
    for batch, record in by_batch.items():
        expected_gpus = 4 if batch == 64 else 8
        expected_accumulation = batch // (expected_gpus * 16)
        expected_targets = [1, 2, 4, 8, 16, 24] if model == "474m" else [1, 2, 4, 8, 16, 32, 48]
        expected_increment = 8 if model == "474m" else 16
        checks = {
            "variant": record.get("variant") == "DR+WT+EmbedWD",
            "lr": str(record.get("lr")) == "2e-3",
            "rank microbatch": record.get("rankMicrobatchSequences") == 16,
            "GPU count": record.get("gpuCount") == expected_gpus,
            "gradient accumulation": record.get("gradientAccumulation") == expected_accumulation,
            "dynamic repacking": record.get("dynamicRepacking") is True,
            "data order": record.get("dataOrder") == "dynamic_repacking_from_e2",
            "weight tying": record.get("weightTying") is True,
            "embedding WD": record.get("decayEmbeddings") is True,
            "targets": record.get("initialTargets") == expected_targets,
            "increment": record.get("epochIncrement") == expected_increment,
            "retry count": record.get("automaticTaskRetries") == 8,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise ValueError(f"{model} BS{batch}: invalid {', '.join(failed)}")
        ladder = [str(value) for value in record["wdLadder"]]
        if not ladder or max(Decimal(value) for value in ladder) > Decimal("1.0"):
            raise ValueError(f"{model} BS{batch}: WD ladder exceeds the 1.0 cap")
        active_wds = [Decimal(str(value)) for value in record.get("activeWds", [])]
        if active_wds and max(active_wds) > Decimal("1.0"):
            raise ValueError(f"{model} BS{batch}: active WD exceeds the 1.0 cap")
        for epoch, results in record.get("results", {}).items():
            if any(Decimal(str(wd)) > Decimal("1.0") for wd in results):
                raise ValueError(f"{model} BS{batch} E{epoch}: result WD exceeds 1.0")
        transition = record.get("policyTransition") or {}
        if transition:
            locked_wd = Decimal(str(transition.get("lockedWd")))
            if locked_wd > Decimal("1.0"):
                raise ValueError(f"{model} BS{batch}: transition WD exceeds 1.0")
            if transition.get("status") == "awaiting-current-stage":
                awaited = transition.get("awaitStage") or {}
                if awaited.get("epoch") is None or awaited.get("wd") is None:
                    raise ValueError(f"{model} BS{batch}: incomplete transition boundary")
        if record.get("policy") in POLICIES:
            locked_wd = str(record.get("lockedWd"))
            finalizer = record.get("policy") == FINALIZER_POLICY
            policy_checks = {
                "WD tuning stopped": record.get("wdTuningStopped") is True,
                "locked WD": locked_wd in ladder,
                "phase comparison": record.get("comparisonPolicy")
                == ("post_decay_only" if finalizer else "within_phase_only"),
                "three decay sources": record.get("postDecaySourceCount") == 3,
                "historical boundary": int(record.get("historicalPreDecayThroughEpoch", 0)) >= 3,
                "pre-decay criterion": finalizer
                or record.get("preDecaySaturationCriterion") == "strict_non_improvement",
                "post-decay criterion": record.get("postDecaySaturationCriterion")
                == "strict_non_improvement",
                "pre-decay disabled": not finalizer
                or record.get("preDecayEvaluation") is False,
                "active WD lock": all(
                    str(value) == locked_wd for value in record.get("activeWds", [])
                ),
            }
            failed_policy = [name for name, valid in policy_checks.items() if not valid]
            if failed_policy:
                raise ValueError(
                    f"{model} BS{batch}: invalid locked-WD policy {', '.join(failed_policy)}"
                )
            for result in record.get("preDecayResults", {}).values():
                if result.get("comparisonGroup") != "pre_decay":
                    raise ValueError(f"{model} BS{batch}: mixed pre-decay comparison group")
            for result in record.get("postDecayResults", {}).values():
                if result.get("comparisonGroup") != "post_decay":
                    raise ValueError(f"{model} BS{batch}: mixed post-decay comparison group")
            selection = record.get("postDecaySelection")
            if selection:
                if not finalizer and selection.get("preDecayDecisionGroup") != "pre_decay":
                    raise ValueError(f"{model} BS{batch}: invalid saturation comparison group")
                if selection.get("postDecaySelectionGroup") != "post_decay":
                    raise ValueError(f"{model} BS{batch}: invalid decay selection group")
                source_count = len(selection.get("postDecaySourceEpochs", []))
                if source_count != 3 and not (finalizer and 1 <= source_count <= 3):
                    raise ValueError(f"{model} BS{batch}: decay selection is not three-way")
                if selection.get("postDecayDecisionGroup", "post_decay") != "post_decay":
                    raise ValueError(f"{model} BS{batch}: invalid post-decay decision group")
                if selection.get("postDecaySaturationCriterion") != "strict_non_improvement":
                    raise ValueError(f"{model} BS{batch}: invalid post-decay criterion")
                if not finalizer and selection.get("postDecaySaturated") is not True:
                    raise ValueError(f"{model} BS{batch}: final selection is not saturated")
            for epoch, decision in record.get("postDecayContinuations", {}).items():
                if decision.get("status") != "continue":
                    raise ValueError(f"{model} BS{batch} E{epoch}: invalid continuation status")
                if decision.get("postDecayDecisionGroup") != "post_decay":
                    raise ValueError(f"{model} BS{batch} E{epoch}: mixed continuation group")
                if decision.get("postDecaySaturated") is not False:
                    raise ValueError(f"{model} BS{batch} E{epoch}: saturated continuation")
                if len(decision.get("postDecaySourceEpochs", [])) != 3:
                    raise ValueError(f"{model} BS{batch} E{epoch}: continuation is not three-way")
        center = str(record["baselineOptimalWd"])
        center_index = ladder.index(center)
        if record.get("initialWds") != ladder[center_index - 1 : center_index + 2]:
            raise ValueError(f"{model} BS{batch}: initial WD neighbors are invalid")
        outputs = record.get("outputByWd", {})
        if sorted(outputs) != sorted(ladder):
            raise ValueError(f"{model} BS{batch}: output map does not cover the WD ladder")
        if len(set(outputs.values())) != len(ladder):
            raise ValueError(f"{model} BS{batch}: WD trajectories share an output directory")
        for wd, output in outputs.items():
            expected_suffix = f"/bs{batch}_dr_wt_embwd_lr2e-3_wd{wd}"
            if not str(output).endswith(expected_suffix):
                raise ValueError(f"{model} BS{batch}: noncanonical output {output}")
        experiment = str(record.get("experiment", ""))
        if len(experiment) != 26:
            raise ValueError(f"{model} BS{batch}: missing Beaker experiment")
        if experiment in experiments:
            raise ValueError(f"{model}: duplicate Beaker experiment {experiment}")
        experiments.add(experiment)


def main() -> None:
    for model in MODELS:
        validate_model(model)
    print("small-dense DR+WT+EmbedWD chains validated")


if __name__ == "__main__":
    main()
