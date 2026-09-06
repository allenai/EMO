#!/usr/bin/env python3
"""Repeat selected Dense-1B WSD decays without mutating the original POST runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_1b_dr_wt_embedwd_grid as dense1b

POLICY = "dense_1b_dr_wt_embwd_redecay_retry01_v1"
RETRY = "retry01"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_run(manifest_path: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("policy") != POLICY or float(manifest.get("decayFraction", -1)) != 0.1:
        raise ValueError("re-decay manifest must preserve the authorized 10% WSD policy")
    matches = [item for item in manifest.get("runs", []) if item.get("id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one re-decay run {run_id}")
    item = matches[0]
    validate_item(item)
    return manifest, item


def validate_item(item: dict[str, Any]) -> None:
    batch = int(item["batchSequences"])
    epoch = int(item["epoch"])
    if batch not in {128, 256}:
        raise ValueError("only the authorized BS128 and BS256 repeats are allowed")
    if str(item["learningRate"]) != "1e-3":
        raise ValueError("the repeats must use LR1e-3")
    expected_wd = "0.3" if batch == 128 else "1.0"
    if str(item["weightDecay"]) != expected_wd:
        raise ValueError(f"BS{batch} repeat must use WD{expected_wd}")
    if (batch, epoch) not in {(128, 24), (256, 36), (256, 40)}:
        raise ValueError("unauthorized re-decay coordinate")
    expected_source = dense1b.stable_step(epoch, batch)
    expected_endpoint = dense1b.total_step(epoch, batch)
    if int(item["sourceStep"]) != expected_source:
        raise ValueError(f"E{epoch} source must be exact step{expected_source}")
    if int(item["endpointStep"]) != expected_endpoint:
        raise ValueError(f"E{epoch} 10% decay endpoint must be step{expected_endpoint}")
    root = Path(str(item["coordinateOutput"]))
    if Path(str(item["sourceCheckpoint"])) != root / f"step{expected_source}":
        raise ValueError("source checkpoint is outside the exact coordinate root")
    expected_output = root / ".all_postdecay_policy" / "post_decay_runs" / f"e{epoch}-{RETRY}"
    if Path(str(item["retryOutput"])) != expected_output:
        raise ValueError("retry output must use the isolated e{epoch}-retry01 namespace")
    if int(item["rankMicrobatchSequences"]) != 8:
        raise ValueError("rank microbatch must remain eight sequences")
    if int(item["gradientAccumulation"]) != batch // 64:
        raise ValueError("gradient accumulation does not reproduce the original global batch")
    if int(item["warmupSteps"]) != 24576 // batch:
        raise ValueError("warmup does not reproduce the original schedule")


def dense_config(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "globalSequences": int(item["batchSequences"]),
        "nprocPerNode": 8,
        "rankMicrobatchSequences": int(item["rankMicrobatchSequences"]),
        "gradientAccumulation": int(item["gradientAccumulation"]),
        "warmupSteps": int(item["warmupSteps"]),
        "coordinates": [
            {
                "lr": str(item["learningRate"]),
                "wd": str(item["weightDecay"]),
                "output": str(item["coordinateOutput"]),
            }
        ],
        "variant": "DR+WT+EmbedWD",
        "runSuffix": RETRY,
    }


def result_path(item: dict[str, Any]) -> Path:
    root = Path(str(item["coordinateOutput"])) / ".all_postdecay_policy"
    return root / "post_decay_retries" / f"e{item['epoch']}-{RETRY}.result.json"


def run_epoch(item: dict[str, Any]) -> dict[str, Any]:
    stored = result_path(item)
    if stored.is_file():
        result = json.loads(stored.read_text())
        print(f"DENSE1B_REDECAY_RESULT id={item['id']} json={json.dumps(result, separators=(',', ':'))}", flush=True)
        return result

    config = dense_config(item)
    epoch = int(item["epoch"])
    source = Path(str(item["sourceCheckpoint"]))
    output = Path(str(item["retryOutput"]))
    raw_endpoint = output / f"step{item['endpointStep']}"
    endpoint = output / f"step{item['endpointStep']}-{RETRY}"
    if not source.is_dir():
        raise FileNotFoundError(f"missing exact retained source {source}")
    if output.parent.name != "post_decay_runs" or not output.name.endswith(f"-{RETRY}"):
        raise RuntimeError("refusing non-isolated retry output")

    name = f"dense-1b-bs{item['batchSequences']}-dr-wt-embwd-e{epoch}-redecay-{RETRY}"
    log_path = (
        Path(str(item["coordinateOutput"]))
        / ".all_postdecay_policy"
        / "logs"
        / f"post_decay_train_e{epoch}-{RETRY}.log"
    )
    if endpoint.is_dir():
        eval_name = name + "-recovered-eval"
        dense1b.run_torch(
            config,
            eval_name,
            dense1b.evaluation_arguments(config, str(item["learningRate"]), str(item["weightDecay"]), endpoint, output / "recovered_eval", eval_name),
            log_path,
        )
    elif raw_endpoint.is_dir():
        eval_name = name + "-recovered-eval"
        dense1b.run_torch(
            config,
            eval_name,
            dense1b.evaluation_arguments(config, str(item["learningRate"]), str(item["weightDecay"]), raw_endpoint, output / "recovered_eval", eval_name),
            log_path,
        )
    else:
        arguments = dense1b.postdecay_training_arguments(
            config,
            str(item["learningRate"]),
            str(item["weightDecay"]),
            epoch,
            source,
            output,
            name,
        )
        print(
            f"DENSE1B_REDECAY_START id={item['id']} batch={item['batchSequences']} epoch={epoch} "
            f"source={source} output={output} endpoint={endpoint}",
            flush=True,
        )
        dense1b.run_torch(config, name, arguments, log_path)

    if raw_endpoint.is_dir() and not endpoint.exists():
        raw_endpoint.rename(endpoint)
    if not endpoint.is_dir():
        raise RuntimeError(f"repeat E{epoch} exited without isolated endpoint {endpoint}")
    result = dense1b.parse_validation(log_path, epoch, "post_decay", endpoint)
    if not math.isfinite(float(result["validationExact"])):
        raise RuntimeError("repeat POST validation is non-finite")
    result.update(
        {
            "policy": POLICY,
            "retry": RETRY,
            "lr": str(item["learningRate"]),
            "wd": str(item["weightDecay"]),
            "batchSequences": int(item["batchSequences"]),
            "decayFraction": 0.1,
            "variant": "DR+WT+EmbedWD",
            "dynamicRepacking": True,
            "weightTying": True,
            "decayEmbeddings": True,
            "sourcePreDecayCheckpoint": str(source),
            "output": str(output),
            "originalValidationExact": float(item["originalValidationExact"]),
            "source": "isolated_wsd_redecay_and_matched_heldout_eval",
        }
    )
    atomic_json(stored, result)
    print(
        f"DENSE1B_REDECAY_RESULT id={item['id']} json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    _, item = load_run(args.manifest, args.run_id)
    if args.validate_only:
        print(f"validated {args.run_id}")
        return
    run_epoch(item)


if __name__ == "__main__":
    main()
