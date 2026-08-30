#!/usr/bin/env python3
"""Run an isolated throughput-recovery producer from a clean Pool-3B E1 checkpoint.

This is deliberately separate from the canonical producer output.  A recovery
attempt reads the canonical clean E1 checkpoint, resets only data-loader state,
and writes all future checkpoints under an attempt-specific directory.  This
allows a replacement node to be benchmarked while the original producer keeps
running, without concurrent writes to either checkpoint lineage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_producer as producer


ALLOWED_COORDINATES = {
    "dense-474m-dclm3b-bs256-lr2e-3-wd0.333",
    "dense-153m-dclm3b-bs256-lr2e-3-wd0.033",
    "dense-153m-dclm3b-bs256-lr2e-3-wd0.1",
}


def recovery_output(item: dict[str, Any], attempt: int) -> Path:
    if attempt < 1:
        raise ValueError("attempt must be positive")
    canonical = Path(str(item["output"]))
    return canonical.with_name(f"{canonical.name}_throughput_recovery_r{attempt}")


def recovery_source(item: dict[str, Any]) -> Path:
    batch = int(item["batchSequences"])
    step = producer.stable_step(1, producer.TARGET_POOL_TOKENS, batch)
    return Path(str(item["output"])) / f"step{step}"


def validate_recovery_source(item: dict[str, Any], *, check_filesystem: bool) -> Path:
    source = recovery_source(item)
    if source.parent != Path(str(item["output"])):
        raise RuntimeError("recovery source must be the canonical Pool-3B output")
    if check_filesystem:
        if not source.is_dir() or not (source / "config.json").is_file():
            raise FileNotFoundError(f"missing clean Pool-3B E1 checkpoint {source}")
        if not (source / "model_and_optim").is_dir():
            raise FileNotFoundError(f"incomplete model/optimizer state at {source}")
        value = json.loads((source / "config.json").read_text())
        expected_batch = int(item["batchSequences"]) * producer.SEQUENCE_LENGTH
        checks = {
            "global_batch_size": (
                int(value["data_loader"]["global_batch_size"]),
                expected_batch,
            ),
            "tie_embeddings": (bool(value["model"]["tie_embeddings"]), True),
            "dynamic_repacking": (
                bool(value["dataset"]["dynamic_repacking"]),
                False,
            ),
        }
        mismatches = [
            name for name, (actual, expected) in checks.items() if actual != expected
        ]
        if mismatches:
            raise RuntimeError(f"Pool-3B E1 recovery-source mismatch: {mismatches}")
    return source


def recovery_arguments(
    config: dict[str, Any], item: dict[str, Any], attempt: int
) -> list[str]:
    output = recovery_output(item, attempt)
    source = recovery_source(item)
    epochs = producer.target_epochs(item, int(config["maxEpoch"]))
    fixed_steps = [
        producer.stable_step(
            epoch,
            producer.TARGET_POOL_TOKENS,
            int(item["batchSequences"]),
        )
        for epoch in epochs
    ]
    run_name = f"{item['id']}-throughput-recovery-r{attempt}"
    return [
        *producer.base_arguments(item, str(config["repeatedManifest"])),
        "--dynamic-repacking",
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {fixed_steps[-1]}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={run_name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            f"dense-{item['model']},dr_wt_embwd,bs256,constant-lr,pre-decay,"
            f"throughput-recovery,r{attempt}]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps="
        + json.dumps(fixed_steps, separators=(",", ":")),
        f"--trainer.load_path={source}",
    ]


def run(config: dict[str, Any], item: dict[str, Any], attempt: int) -> None:
    producer.validate_coordinate(config, item, check_filesystem=True)
    source = validate_recovery_source(item, check_filesystem=True)
    output = recovery_output(item, attempt)
    if output == Path(str(item["output"])):
        raise RuntimeError("recovery output must not equal the canonical output")
    existing_steps = list(output.glob("step*")) if output.is_dir() else []
    if existing_steps:
        raise RuntimeError(f"recovery output already contains checkpoints: {existing_steps}")
    state_dir = output / ".throughput_recovery"
    producer.atomic_json(
        state_dir / "recovery.json",
        {
            "policy": "dense_small_pool3b_throughput_recovery_v1",
            "coordinate": item["id"],
            "attempt": attempt,
            "status": "running",
            "sourceCheckpoint": str(source),
            "output": str(output),
            "canonicalOutputUntouched": True,
        },
    )
    run_name = f"{item['id']}-throughput-recovery-r{attempt}"
    print(
        f"DENSE_SMALL_POOL3B_THROUGHPUT_RECOVERY_START id={item['id']} "
        f"attempt={attempt} source={source} output={output} "
        "dynamic_repacking=true reset_data_loader=true evaluation=false",
        flush=True,
    )
    producer.run_torch(
        run_name,
        recovery_arguments(config, item, attempt),
        state_dir / "producer.log",
    )
    print(
        f"DENSE_SMALL_POOL3B_THROUGHPUT_RECOVERY_COMPLETE id={item['id']} "
        f"attempt={attempt} output={output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate", choices=sorted(ALLOWED_COORDINATES), required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = producer.load_manifest(args.manifest)
    item = producer.coordinate(config, args.coordinate)
    producer.validate_coordinate(config, item, check_filesystem=not args.validate_only)
    validate_recovery_source(item, check_filesystem=not args.validate_only)
    output = recovery_output(item, args.attempt)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "coordinate": item["id"],
                    "attempt": args.attempt,
                    "source": str(recovery_source(item)),
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return
    run(config, item, args.attempt)


if __name__ == "__main__":
    main()
