#!/usr/bin/env python3
"""Run isolated E1/E2/E4 WSD frontier evaluations for Dense-1B DR+WT+EmbedWD."""

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

POLICY = "dense_1b_dr_wt_embwd_early_frontier_v1"
RUN_SUFFIX = "early_frontier_v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_item(item: dict[str, Any]) -> None:
    batch = int(item["batchSequences"])
    epoch = int(item["epoch"])
    if batch not in {128, 256} or epoch not in {1, 2, 4}:
        raise ValueError("only BS128/BS256 E1/E2/E4 frontier evaluations are authorized")
    if str(item["learningRate"]) != "1e-3" or str(item["weightDecay"]) != "0.3":
        raise ValueError("frontier evaluations must use LR1e-3/WD0.3")
    source_step = dense1b.stable_step(epoch, batch)
    endpoint_step = dense1b.total_step(epoch, batch)
    if int(item["sourceStep"]) != source_step or int(item["endpointStep"]) != endpoint_step:
        raise ValueError(f"BS{batch} E{epoch} does not use the exact 10% WSD steps")
    root = Path(str(item["coordinateOutput"]))
    if Path(str(item["sourceCheckpoint"])) != root / f"step{source_step}":
        raise ValueError("source checkpoint is outside the canonical coordinate root")
    expected_output = root / ".early_frontier_eval_v1" / "post_decay_runs" / f"e{epoch}"
    if Path(str(item["evaluationOutput"])) != expected_output:
        raise ValueError("evaluation output is not in the isolated early-frontier namespace")
    if int(item["rankMicrobatchSequences"]) != 8:
        raise ValueError("rank microbatch must remain eight sequences")
    if int(item["gradientAccumulation"]) != batch // 64:
        raise ValueError("gradient accumulation does not reproduce the original global batch")
    if int(item["warmupSteps"]) != 24576 // batch:
        raise ValueError("warmup does not reproduce the original schedule")


def load_runs(manifest_path: Path, batch: int) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("policy") != POLICY or float(manifest.get("decayFraction", -1)) != 0.1:
        raise ValueError("manifest must preserve the authorized 10% WSD policy")
    runs = [item for item in manifest.get("runs", []) if int(item["batchSequences"]) == batch]
    if [int(item["epoch"]) for item in runs] != [1, 2, 4]:
        raise ValueError(f"BS{batch} must contain exactly E1, E2, and E4 in order")
    for item in runs:
        validate_item(item)
    return runs


def dense_config(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "globalSequences": int(item["batchSequences"]),
        "nprocPerNode": 8,
        "rankMicrobatchSequences": int(item["rankMicrobatchSequences"]),
        "gradientAccumulation": int(item["gradientAccumulation"]),
        "warmupSteps": int(item["warmupSteps"]),
        "coordinates": [{"lr": "1e-3", "wd": "0.3", "output": item["coordinateOutput"]}],
        "variant": "DR+WT+EmbedWD",
        "runSuffix": RUN_SUFFIX,
    }


def result_path(item: dict[str, Any]) -> Path:
    return (
        Path(str(item["coordinateOutput"]))
        / ".early_frontier_eval_v1"
        / "results"
        / f"e{item['epoch']}.result.json"
    )


def run_epoch(item: dict[str, Any]) -> dict[str, Any]:
    stored = result_path(item)
    if stored.is_file():
        result = json.loads(stored.read_text())
        print(f"DENSE1B_EARLY_FRONTIER_RESULT id={item['id']} json={json.dumps(result, separators=(',', ':'))}", flush=True)
        return result
    config = dense_config(item)
    epoch = int(item["epoch"])
    source = Path(str(item["sourceCheckpoint"]))
    output = Path(str(item["evaluationOutput"]))
    endpoint = output / f"step{item['endpointStep']}"
    if not source.is_dir():
        raise FileNotFoundError(f"missing exact retained source {source}")
    if output.parent.name != "post_decay_runs" or output.parent.parent.name != ".early_frontier_eval_v1":
        raise RuntimeError("refusing non-isolated early-frontier output")
    name = f"dense-1b-bs{item['batchSequences']}-dr-wt-embwd-e{epoch}-early-frontier-v1"
    log_path = Path(str(item["coordinateOutput"])) / ".early_frontier_eval_v1" / "logs" / f"e{epoch}.log"
    if endpoint.is_dir():
        eval_name = name + "-recovered-eval"
        dense1b.run_torch(
            config,
            eval_name,
            dense1b.evaluation_arguments(config, "1e-3", "0.3", endpoint, output / "recovered_eval", eval_name),
            log_path,
        )
    else:
        print(
            f"DENSE1B_EARLY_FRONTIER_START id={item['id']} batch={item['batchSequences']} epoch={epoch} "
            f"source={source} output={output} endpoint={endpoint}",
            flush=True,
        )
        dense1b.run_torch(
            config,
            name,
            dense1b.postdecay_training_arguments(config, "1e-3", "0.3", epoch, source, output, name),
            log_path,
        )
    if not endpoint.is_dir():
        raise RuntimeError(f"BS{item['batchSequences']} E{epoch} exited without exact endpoint {endpoint}")
    result = dense1b.parse_validation(log_path, epoch, "post_decay", endpoint)
    if not math.isfinite(float(result["validationExact"])):
        raise RuntimeError("POST validation is non-finite")
    result.update(
        {
            "policy": POLICY,
            "lr": "1e-3",
            "wd": "0.3",
            "batchSequences": int(item["batchSequences"]),
            "decayFraction": 0.1,
            "variant": "DR+WT+EmbedWD",
            "dynamicRepacking": epoch > 1,
            "weightTying": True,
            "decayEmbeddings": True,
            "sourcePreDecayCheckpoint": str(source),
            "output": str(output),
            "source": "isolated_early_frontier_wsd_and_matched_heldout_eval",
        }
    )
    atomic_json(stored, result)
    print(f"DENSE1B_EARLY_FRONTIER_RESULT id={item['id']} json={json.dumps(result, separators=(',', ':'), sort_keys=True)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(128, 256), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.manifest, args.batch)
    if args.validate_only:
        print(f"validated BS{args.batch} E1/E2/E4")
        return
    for item in runs:
        run_epoch(item)


if __name__ == "__main__":
    main()
