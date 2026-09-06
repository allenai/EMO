#!/usr/bin/env python3
"""Compare matched E20 POSTs, then conditionally launch LR1e-3/WD0.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_dense_153m_bs32_lr_wd_hybrid as hybrid

POLICY = "dense_153m_bs32_e20_lr_gate_v1"
EPOCH = 20
ENDPOINT_STEP = 152588
STATE = ".dense_153m_bs32_e20_lr_gate_v1"


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = {
        "policy": POLICY,
        "epoch": EPOCH,
        "followupLearningRate": "1e-3",
        "followupWeightDecay": "0.1",
        "followupEvaluationEpochs": [40, 48, 56, 64, 72, 80],
        "decayFraction": 0.1,
    }
    bad = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if bad:
        raise ValueError(f"fixed-policy manifest mismatch: {bad}")
    for label, lr in (("probe", "5e-4"),):
        item = value[label]
        if item.get("learningRate") != lr or item.get("weightDecay") != "0.033":
            raise ValueError(f"invalid {label} coordinate")
        if not str(item.get("sourceCheckpoint", "")).endswith("/step137328"):
            raise ValueError(f"{label} must use exact E20 PD step137328")
        output = Path(str(item.get("output", "")))
        allowed_root = "/weka/oe-training-default/sewonm/icsl/models/"
        if not output.is_absolute() or ".." in output.parts or not str(output).startswith(allowed_root):
            raise ValueError(f"invalid {label} output")
    baseline = value.get("baselineReference", {})
    if baseline != {
        "epoch": 16,
        "learningRate": "1e-3",
        "weightDecay": "0.033",
        "validationExact": 3.4,
    }:
        raise ValueError("baseline must be the completed LR1e-3/WD0.033 E16 POST reference")
    if hybrid.stable_step(EPOCH) != 137328 or hybrid.total_step(EPOCH) != ENDPOINT_STEP:
        raise RuntimeError("E20 step calculation changed")
    return value


def evaluate(value: dict[str, Any], label: str) -> dict[str, Any]:
    item = value[label]
    source = hybrid.validate_predecay(
        Path(item["sourceCheckpoint"]), item["learningRate"], item["weightDecay"]
    )
    output = Path(item["output"])
    state = output / STATE
    owner = state / "owner.json"
    result_path = state / "result.json"
    expected_owner = {
        "policy": POLICY,
        "id": value["id"],
        "label": label,
        "sourceCheckpoint": str(source),
        "output": str(output),
    }
    if owner.is_file() and json.loads(owner.read_text()) != expected_owner:
        raise RuntimeError(f"output ownership mismatch at {owner}")
    hybrid.atomic_json(owner, expected_owner)
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        print(f"DENSE153M_BS32_E20_GATE_RESULT label={label} json=" + json.dumps(result, separators=(",", ":")), flush=True)
        return result
    endpoint = output / f"step{ENDPOINT_STEP}"
    log_path = state / "post.log"
    recovery_log = state / "recovered_eval.log"
    if not hybrid.checkpoint_complete(endpoint):
        if output.exists() and any(path.name != STATE for path in output.iterdir()):
            hybrid.quarantine_partial(output)
            state = output / STATE
            owner = state / "owner.json"
            result_path = state / "result.json"
            log_path = state / "post.log"
            recovery_log = state / "recovered_eval.log"
            hybrid.atomic_json(owner, expected_owner)
        name, arguments = hybrid.post_arguments(
            {"runSuffix": value["runSuffix"]}, label, item["learningRate"],
            item["weightDecay"], EPOCH, source, output,
        )
        print(f"DENSE153M_BS32_E20_GATE_POST_START label={label} source={source} output={output}", flush=True)
        hybrid.run_torch(name, arguments, log_path)
    if not hybrid.checkpoint_complete(endpoint):
        raise RuntimeError(f"{label} E20 decay exited without complete endpoint {endpoint}")
    try:
        result = hybrid.parse_result([log_path], EPOCH)
    except RuntimeError:
        recovered_name = f"dense_153m_bs32_e20_{label}_recovered_eval_{value['runSuffix']}"
        hybrid.run_torch(
            recovered_name,
            hybrid.recovery_eval_arguments(item["learningRate"], item["weightDecay"], endpoint, output / "recovered_eval", recovered_name),
            recovery_log,
        )
        result = hybrid.parse_result([log_path, recovery_log], EPOCH)
    result.update({
        "policy": POLICY, "label": label, "lr": item["learningRate"],
        "wd": item["weightDecay"], "preDecayCheckpoint": str(source),
        "endpointCheckpoint": str(endpoint), "postOutput": str(output),
        "decayFraction": 0.1,
    })
    hybrid.atomic_json(result_path, result)
    print(f"DENSE153M_BS32_E20_GATE_RESULT label={label} json=" + json.dumps(result, separators=(",", ":")), flush=True)
    return result


def run(value: dict[str, Any]) -> None:
    probe = evaluate(value, "probe")
    probe_value = float(probe["validationExact"])
    baseline = value["baselineReference"]
    baseline_value = float(baseline["validationExact"])
    selected = "1e-3" if probe_value > baseline_value else "5e-4"
    decision = {
        "policy": POLICY,
        "selectedLearningRate": selected,
        "selectedWeightDecay": "0.033",
        "reason": "probe_e20_strictly_worse_than_lr1e3_e16_reference" if selected == "1e-3" else "probe_e20_not_strictly_worse_than_lr1e3_e16_reference",
        "probeEpoch": 20,
        "baselineEpoch": int(baseline["epoch"]),
        "probeValidationExact": probe_value,
        "baselineValidationExact": baseline_value,
    }
    print("DENSE153M_BS32_HYBRID_LR_DECISION json=" + json.dumps(decision, separators=(",", ":"), sort_keys=True), flush=True)
    if selected != "1e-3":
        print("DENSE153M_BS32_E20_GATE_COMPLETE action=hold_no_followup", flush=True)
        return

    followup = hybrid.load_manifest(Path(value["followupManifest"]))
    lr, wd = "1e-3", "0.1"
    hybrid.claim_output(followup, "followup", lr, wd)
    print(f"DENSE153M_BS32_HYBRID_FOLLOWUP_START lr={lr} wd={wd} output={hybrid.coordinate_output(followup, 'followup', lr)}", flush=True)
    workflow_state: dict[str, Any] = {"policy": POLICY, "id": value["id"], "phase": "followup"}
    for epoch in value["followupEvaluationEpochs"]:
        hybrid.ensure_predecay(followup, "followup", lr, wd, int(epoch), workflow_state)
        hybrid.evaluate(followup, "followup", lr, wd, int(epoch), workflow_state)
    print(f"DENSE153M_BS32_HYBRID_WORKFLOW_COMPLETE selectedLearningRate={lr} followupWeightDecay={wd}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    value = load_manifest(args.manifest)
    if args.validate_only:
        print(json.dumps(value, indent=2))
        return
    run(value)


if __name__ == "__main__":
    main()
