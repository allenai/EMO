#!/usr/bin/env python3
"""Evaluate only the user-requested POST endpoints and then stop the chain."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_small_dense_dr_wt_embedwd_chain as adaptive
import run_small_dense_locked_wd_predecay_chain as locked

POLICY = "locked_wd_requested_postdecay_finalizer_v1"
SOURCE_COUNT = 3


def parse_epochs(value: str) -> list[int]:
    if not value.strip():
        return []
    epochs = [int(item) for item in value.split(",")]
    if len(epochs) != len(set(epochs)) or epochs != sorted(epochs):
        raise ValueError("--epochs must be unique and strictly increasing")
    return epochs


def load_post_results(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    directory = locked.policy_state_dir(config) / "post_decay"
    results: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("e*.result.json") if directory.is_dir() else []):
        result = json.loads(path.read_text())
        if result.get("comparisonGroup") != "post_decay":
            raise RuntimeError(f"mixed comparison group in {path}")
        results[int(result["epoch"])] = result
    return results


def saturation_at(
    results: dict[int, dict[str, Any]], decision_epoch: int
) -> bool:
    epochs = [epoch for epoch in sorted(results) if epoch <= decision_epoch]
    if len(epochs) < SOURCE_COUNT or epochs[-1] != decision_epoch:
        return False
    previous, current = results[epochs[-2]], results[epochs[-1]]
    return Decimal(str(current["validationExact"])) >= Decimal(
        str(previous["validationExact"])
    )


def decision(
    config: dict[str, Any],
    requested: list[int],
    results: dict[int, dict[str, Any]],
    *,
    saturated: bool,
    preserve_existing_selection: bool,
) -> dict[str, Any]:
    decision_epoch = requested[-1] if requested else max(results)
    available = [epoch for epoch in sorted(results) if epoch <= decision_epoch]
    sources = available[-SOURCE_COUNT:]
    selected_epoch = min(
        sources,
        key=lambda epoch: (
            Decimal(str(results[epoch]["validationExact"])),
            epoch,
        ),
    )
    selected = results[selected_epoch]
    return {
        "status": "complete" if saturated else "stopped_by_user",
        "policy": POLICY,
        "lockedWd": str(config["lockedWd"]),
        "terminationReason": (
            "post_decay_non_improvement"
            if saturated
            else "user_requested_frontier_stop"
        ),
        "requestedPostDecayEpochs": requested,
        "evaluatedThroughEpoch": decision_epoch,
        "postDecayDecisionGroup": "post_decay",
        "postDecaySelectionGroup": "post_decay",
        "postDecaySaturationCriterion": "strict_non_improvement",
        "postDecaySaturated": saturated,
        "postDecaySourceEpochs": sources,
        "postDecayValidationExact": {
            str(epoch): float(results[epoch]["validationExact"]) for epoch in sources
        },
        "selectedPostDecayEpoch": selected_epoch,
        "selectedPostDecayValidationExact": float(selected["validationExact"]),
        "selectedCheckpoint": selected["checkpoint"],
        "preserveExistingSelection": preserve_existing_selection,
    }


def emit(config: dict[str, Any], value: dict[str, Any]) -> None:
    print(
        f"SMALL_POSTDECAY_FINALIZER_COMPLETE model={config['model']} "
        f"bs={config['globalSequences']} status={value['status']} "
        f"json={json.dumps(value, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )


def run(
    config: dict[str, Any],
    requested: list[int],
    *,
    stop_on_saturation: bool,
    preserve_existing_selection: bool,
) -> dict[str, Any]:
    locked.validate_config(config)
    for epoch in requested:
        adaptive.targets_through(config, epoch)
        if epoch < int(config["historicalPreDecayStartEpoch"]):
            raise ValueError(f"E{epoch} predates the in-policy historical boundary")
    results = load_post_results(config)
    evaluated: list[int] = []
    for epoch in requested:
        results[epoch] = locked.run_postdecay(config, epoch)
        evaluated.append(epoch)
        if stop_on_saturation and saturation_at(results, epoch):
            value = decision(
                config,
                evaluated,
                results,
                saturated=True,
                preserve_existing_selection=preserve_existing_selection,
            )
            locked.atomic_json(
                locked.policy_state_dir(config) / "requested_finalization.json", value
            )
            if not preserve_existing_selection:
                locked.atomic_json(locked.selection_path(config), value)
            emit(config, value)
            return value
    if not results:
        raise RuntimeError("finalizer has no completed POST result")
    decision_epoch = evaluated[-1] if evaluated else max(results)
    value = decision(
        config,
        evaluated,
        results,
        saturated=saturation_at(results, decision_epoch),
        preserve_existing_selection=preserve_existing_selection,
    )
    locked.atomic_json(
        locked.policy_state_dir(config) / "requested_finalization.json", value
    )
    if not preserve_existing_selection:
        locked.atomic_json(locked.selection_path(config), value)
    emit(config, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", default="")
    parser.add_argument("--stop-on-saturation", action="store_true")
    parser.add_argument("--preserve-existing-selection", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    locked.validate_config(config)
    epochs = parse_epochs(args.epochs)
    for epoch in epochs:
        adaptive.targets_through(config, epoch)
    if args.validate_only:
        print(f"validated {args.manifest}: POST epochs {epochs}")
        return
    run(
        config,
        epochs,
        stop_on_saturation=args.stop_on_saturation,
        preserve_existing_selection=args.preserve_existing_selection,
    )


if __name__ == "__main__":
    main()
