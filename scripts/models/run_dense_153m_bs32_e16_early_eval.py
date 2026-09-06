#!/usr/bin/env python3
"""Wait for the exact E16 PD checkpoint, then run an isolated 10% WSD POST."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_dense_153m_bs32_lr_wd_hybrid as hybrid

POLICY = "dense_153m_bs32_e16_early_eval_v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    hybrid.atomic_json(path, value)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = {
        "policy": POLICY,
        "epoch": 16,
        "learningRate": "5e-4",
        "weightDecay": "0.033",
        "sourceCheckpoint": (
            "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm1b/"
            "bs32_lr5e-4_wd0.033_hybrid_v1/constant_lr/step109863"
        ),
        "output": (
            "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm1b/"
            "bs32_lr5e-4_wd0.033_hybrid_v1/post_decay_runs/e16-early-v1"
        ),
        "endpointCheckpoint": (
            "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm1b/"
            "bs32_lr5e-4_wd0.033_hybrid_v1/post_decay_runs/e16-early-v1/step122071"
        ),
    }
    mismatches = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
    if mismatches:
        raise ValueError(f"fixed-policy manifest mismatch: {mismatches}")
    if hybrid.stable_step(16) != 109863 or hybrid.total_step(16) != 122071:
        raise RuntimeError("E16 step calculation changed unexpectedly")
    if int(value.get("waitTimeoutSeconds", 0)) <= 0 or int(value.get("pollSeconds", 0)) <= 0:
        raise ValueError("wait and poll durations must be positive")
    return value


def wait_for_source(value: dict[str, Any]) -> Path:
    source = Path(value["sourceCheckpoint"])
    deadline = time.monotonic() + int(value["waitTimeoutSeconds"])
    while not hybrid.checkpoint_complete(source):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for exact E16 checkpoint {source}")
        print(
            f"DENSE153M_BS32_E16_EARLY_WAIT source={source} "
            f"checkedAt={datetime.now(tz=UTC).isoformat()}",
            flush=True,
        )
        time.sleep(int(value["pollSeconds"]))
    return hybrid.validate_predecay(source, value["learningRate"], value["weightDecay"])


def run(value: dict[str, Any]) -> None:
    source = wait_for_source(value)
    output = Path(value["output"])
    endpoint = Path(value["endpointCheckpoint"])
    state_root = output / ".dense_153m_bs32_e16_early_eval_v1"
    owner = state_root / "owner.json"
    result_path = state_root / "result.json"
    expected_owner = {
        "policy": POLICY,
        "id": value["id"],
        "sourceCheckpoint": str(source),
        "output": str(output),
    }
    if owner.is_file() and json.loads(owner.read_text()) != expected_owner:
        raise RuntimeError(f"output ownership mismatch at {owner}")
    atomic_json(owner, expected_owner)
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        print("DENSE153M_BS32_E16_EARLY_RESULT json=" + json.dumps(result, separators=(",", ":")), flush=True)
        return

    log_path = state_root / "post.log"
    recovery_log = state_root / "recovered_eval.log"
    if not hybrid.checkpoint_complete(endpoint):
        if output.exists() and any(path.name != state_root.name for path in output.iterdir()):
            hybrid.quarantine_partial(output)
            state_root = output / ".dense_153m_bs32_e16_early_eval_v1"
            owner = state_root / "owner.json"
            result_path = state_root / "result.json"
            log_path = state_root / "post.log"
            recovery_log = state_root / "recovered_eval.log"
            atomic_json(owner, expected_owner)
        name, arguments = hybrid.post_arguments(
            {"runSuffix": "sm0905bs32e16earlyv1"},
            "early-e16",
            value["learningRate"],
            value["weightDecay"],
            16,
            source,
            output,
        )
        print(
            f"DENSE153M_BS32_E16_EARLY_START source={source} output={output} endpoint={endpoint}",
            flush=True,
        )
        hybrid.run_torch(name, arguments, log_path)
    if not hybrid.checkpoint_complete(endpoint):
        raise RuntimeError(f"E16 decay exited without complete endpoint {endpoint}")
    try:
        result = hybrid.parse_result([log_path], 16)
    except RuntimeError:
        recovered_name = "dense_153m_bs32_lr5e-4_wd0.033_e16_early_recovered_eval_v1"
        hybrid.run_torch(
            recovered_name,
            hybrid.recovery_eval_arguments(
                value["learningRate"], value["weightDecay"], endpoint,
                output / "recovered_eval", recovered_name,
            ),
            recovery_log,
        )
        result = hybrid.parse_result([log_path, recovery_log], 16)
    result.update(
        {
            "policy": POLICY,
            "lr": value["learningRate"],
            "wd": value["weightDecay"],
            "preDecayCheckpoint": str(source),
            "endpointCheckpoint": str(endpoint),
            "postOutput": str(output),
            "decayFraction": float(Decimal("0.1")),
            "diagnosticOnly": True,
            "source": "isolated_early_e16_wsd_decay_heldout_and_downstream_eval",
        }
    )
    atomic_json(result_path, result)
    print("DENSE153M_BS32_E16_EARLY_RESULT json=" + json.dumps(result, separators=(",", ":")), flush=True)


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
