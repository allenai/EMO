#!/usr/bin/env python3
"""Continue the exact 153M Original BS32 E80 frontier until POST saturation."""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_153m_bs32_lr_wd_hybrid as hybrid

POLICY = "dense_153m_original_bs32_e80_post_saturation_v1"
STATE_DIRECTORY = ".dense_153m_original_bs32_e80_saturation_v1"
EXPECTED_OUTPUT = Path(
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_153m_dclm1b/bs32_lr1e-3_wd0.1_hybrid_v1"
)
EXPECTED_SOURCE = EXPECTED_OUTPUT / "constant_lr" / "step549316"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def state_root() -> Path:
    return EXPECTED_OUTPUT / STATE_DIRECTORY


def validate_manifest(value: dict[str, Any]) -> None:
    expected = {
        "policy": POLICY,
        "id": "dense-153m-original-bs32-e80-saturation-v1",
        "model": "153m",
        "pool": "dclm1b",
        "poolTokens": 1_000_000_000,
        "sequenceLength": 4096,
        "globalSequences": 32,
        "nprocPerNode": 2,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 1,
        "warmupSteps": 768,
        "saveEveryEpochs": 4,
        "output": str(EXPECTED_OUTPUT),
        "constantOutput": str(EXPECTED_OUTPUT / "constant_lr"),
        "initialSourceCheckpoint": str(EXPECTED_SOURCE),
        "previousEpoch": 80,
        "previousValidationExact": 3.351,
        "firstEpoch": 88,
        "epochIncrement": 8,
        "comparisonMetric": "healthy_matched_post_validationExact",
        "saturationCriterion": "strict_non_improvement",
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
    }
    mismatches = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if mismatches:
        raise ValueError(f"manifest fixed-policy mismatch: {mismatches}")
    if Decimal(str(value.get("learningRate"))) != Decimal("0.001"):
        raise ValueError("learning rate must remain 1e-3")
    if Decimal(str(value.get("weightDecay"))) != Decimal("0.1"):
        raise ValueError("weight decay must remain 0.1")


def claim_output(value: dict[str, Any]) -> None:
    if not EXPECTED_OUTPUT.is_dir():
        raise FileNotFoundError(f"canonical trajectory is missing: {EXPECTED_OUTPUT}")
    old_owner = EXPECTED_OUTPUT / hybrid.STATE_DIRECTORY / "owner.json"
    if not old_owner.is_file():
        raise RuntimeError(f"completed hybrid ownership marker is missing: {old_owner}")
    owner = json.loads(old_owner.read_text())
    expected_old = {
        "policy": hybrid.POLICY,
        "workflowId": "dense-153m-bs32-lr-wd-hybrid-v1",
        "phase": "followup",
        "learningRate": "1e-3",
        "weightDecay": "0.1",
        "constantOutput": str(EXPECTED_OUTPUT / "constant_lr"),
    }
    if owner != expected_old:
        raise RuntimeError(f"canonical trajectory ownership mismatch: {owner}")
    marker = state_root() / "owner.json"
    expected_new = {
        "policy": POLICY,
        "workflowId": value["id"],
        "predecessorPolicy": hybrid.POLICY,
        "initialSourceCheckpoint": str(EXPECTED_SOURCE),
        "constantOutput": str(EXPECTED_OUTPUT / "constant_lr"),
    }
    if marker.is_file() and json.loads(marker.read_text()) != expected_new:
        raise RuntimeError(f"continuation ownership mismatch: {marker}")
    atomic_json(marker, expected_new)


def runner_value(value: dict[str, Any]) -> dict[str, Any]:
    return {"runSuffix": value["runSuffix"]}


def ensure_predecay(value: dict[str, Any], epoch: int, state: dict[str, Any]) -> Path:
    lr, wd = str(value["learningRate"]), str(value["weightDecay"])
    target = EXPECTED_OUTPUT / "constant_lr" / f"step{hybrid.stable_step(epoch)}"
    if not hybrid.checkpoint_complete(target):
        source: Path | None = None
        source_epoch: int | None = None
        for candidate_epoch in range(epoch - 4, 79, -4):
            candidate = EXPECTED_OUTPUT / "constant_lr" / f"step{hybrid.stable_step(candidate_epoch)}"
            if hybrid.checkpoint_complete(candidate):
                source_epoch, source = candidate_epoch, hybrid.validate_predecay(candidate, lr, wd)
                break
        if source is None:
            raise FileNotFoundError(f"no complete recovery checkpoint exists before E{epoch}")
        name, arguments = hybrid.constant_arguments(
            runner_value(value), "followup", lr, wd, EXPECTED_OUTPUT, epoch, source
        )
        state.update({"status": "producer_running", "currentEpoch": epoch, "sourceEpoch": source_epoch})
        atomic_json(state_root() / "workflow.json", state)
        print(
            f"DENSE153M_ORIGINAL_BS32_SATURATION_PD_START epoch={epoch} "
            f"sourceEpoch={source_epoch} source={source} output={EXPECTED_OUTPUT / 'constant_lr'}",
            flush=True,
        )
        hybrid.run_torch(name, arguments, state_root() / "logs" / f"pd_e{epoch}.log")
    target = hybrid.validate_predecay(target, lr, wd)
    print(f"DENSE153M_ORIGINAL_BS32_SATURATION_PD_RETAINED epoch={epoch} checkpoint={target}", flush=True)
    return target


def evaluate(value: dict[str, Any], epoch: int, state: dict[str, Any]) -> dict[str, Any]:
    result_path = state_root() / "results" / f"e{epoch}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        emit_result(epoch, result)
        return result
    lr, wd = str(value["learningRate"]), str(value["weightDecay"])
    source = hybrid.validate_predecay(
        EXPECTED_OUTPUT / "constant_lr" / f"step{hybrid.stable_step(epoch)}", lr, wd
    )
    output = EXPECTED_OUTPUT / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{hybrid.total_step(epoch)}"
    log_path = state_root() / "logs" / f"post_e{epoch}.log"
    recovery_log = state_root() / "logs" / f"post_e{epoch}_recovered_eval.log"
    state.update({"status": "post_running", "currentEpoch": epoch})
    atomic_json(state_root() / "workflow.json", state)
    if not hybrid.checkpoint_complete(endpoint):
        if output.exists():
            hybrid.quarantine_partial(output)
        name, arguments = hybrid.post_arguments(
            runner_value(value), "followup", lr, wd, epoch, source, output
        )
        print(
            f"DENSE153M_ORIGINAL_BS32_SATURATION_POST_START epoch={epoch} "
            f"source={source} output={output}",
            flush=True,
        )
        hybrid.run_torch(name, arguments, log_path)
    if not hybrid.checkpoint_complete(endpoint):
        raise RuntimeError(f"E{epoch} decay exited without complete endpoint {endpoint}")
    try:
        result = hybrid.parse_result([log_path], epoch)
    except RuntimeError:
        name = f"dense_153m_original_bs32_post_e{epoch}_recovered_eval_{value['runSuffix']}"
        hybrid.run_torch(
            name,
            hybrid.recovery_eval_arguments(lr, wd, endpoint, output / "recovered_eval", name),
            recovery_log,
        )
        result = hybrid.parse_result([log_path, recovery_log], epoch)
    result.update(
        {
            "policy": POLICY,
            "phase": "saturation_continuation",
            "comparisonGroup": "post_decay",
            "lr": lr,
            "wd": wd,
            "output": str(EXPECTED_OUTPUT),
            "constantOutput": str(EXPECTED_OUTPUT / "constant_lr"),
            "preDecayCheckpoint": str(source),
            "sourcePreDecayCheckpoint": str(source),
            "endpointCheckpoint": str(endpoint),
            "postOutput": str(output),
            "dynamicRepacking": False,
            "weightTying": False,
            "embeddingWeightDecay": "zero",
            "source": "integrated_isolated_wsd_decay_heldout_and_downstream_eval",
        }
    )
    atomic_json(result_path, result)
    emit_result(epoch, result)
    return result


def emit_result(epoch: int, result: dict[str, Any]) -> None:
    print(
        f"DENSE153M_ORIGINAL_BS32_SATURATION_POST_RESULT epoch={epoch} "
        f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )


def run(value: dict[str, Any]) -> None:
    claim_output(value)
    if not hybrid.checkpoint_complete(EXPECTED_SOURCE):
        raise FileNotFoundError(f"exact E80 source is incomplete: {EXPECTED_SOURCE}")
    hybrid.validate_predecay(EXPECTED_SOURCE, "1e-3", "0.1")
    state_path = state_root() / "workflow.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "policy": POLICY,
        "id": value["id"],
        "status": "starting",
        "currentEpoch": 88,
        "previousEpoch": 80,
        "previousValidationExact": 3.351,
        "minRuntimeOmitted": True,
    }
    if state.get("policy") != POLICY or state.get("id") != value["id"]:
        raise RuntimeError("workflow state ownership mismatch")
    atomic_json(state_path, state)
    previous_epoch = int(value["previousEpoch"])
    previous_validation = float(value["previousValidationExact"])
    epoch = int(value["firstEpoch"])
    while True:
        result_path = state_root() / "results" / f"e{epoch}.json"
        decision_path = state_root() / "decisions" / f"e{epoch}.json"
        if result_path.is_file() and decision_path.is_file():
            result = json.loads(result_path.read_text())
            decision = json.loads(decision_path.read_text())
            emit_result(epoch, result)
        else:
            ensure_predecay(value, epoch, state)
            result = evaluate(value, epoch, state)
            action = "continue" if float(result["validationExact"]) < previous_validation else "stop"
            decision = {
                "policy": POLICY,
                "epoch": epoch,
                "previousEpoch": previous_epoch,
                "previousValidationExact": previous_validation,
                "validationExact": float(result["validationExact"]),
                "criterion": "strict_non_improvement",
                "action": action,
            }
            atomic_json(decision_path, decision)
        print(
            "DENSE153M_ORIGINAL_BS32_SATURATION_DECISION json="
            + json.dumps(decision, separators=(",", ":"), sort_keys=True),
            flush=True,
        )
        if decision["action"] == "stop":
            state.update({"status": "saturated", "currentEpoch": epoch, "saturatedEpoch": epoch})
            atomic_json(state_path, state)
            print(f"DENSE153M_ORIGINAL_BS32_SATURATED epoch={epoch}", flush=True)
            return
        previous_epoch = epoch
        previous_validation = float(result["validationExact"])
        epoch += int(value["epochIncrement"])
        state.update(
            {
                "status": "continuing",
                "currentEpoch": epoch,
                "previousEpoch": previous_epoch,
                "previousValidationExact": previous_validation,
            }
        )
        atomic_json(state_path, state)


def main() -> None:
    args = parse_args()
    value = json.loads(args.manifest.read_text())
    validate_manifest(value)
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    run(value)


if __name__ == "__main__":
    main()
