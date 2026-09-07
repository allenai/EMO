#!/usr/bin/env python3
"""Run one fresh exact-WD 474M Pool-3B BS256 probe through E16 and E32."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import run_dense_small_pool3b_checkpoint_evaluator as evaluator
import run_dense_small_pool3b_checkpoint_producer as producer
import run_small_dense_dr_wt_embedwd_chain as original

POLICY = "dense_474m_pool3b_bs256_wd_probes_v1"
EVALUATION_EPOCHS = (16, 32)
STATE_NAME = ".dense_474m_pool3b_bs256_wd_probe_v1"
DEFAULT_MANIFEST = Path(
    "scripts/models/manifests/dense-474m-pool3b-bs256-wd-probes-v1.json"
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(
    path: Path, coordinate_id: str, *, check_filesystem: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    if config.get("policy") != POLICY or config.get("evaluationEpochs") != [16, 32]:
        raise ValueError("unexpected WD-probe policy or evaluation schedule")
    matches = [item for item in config.get("coordinates", []) if item.get("id") == coordinate_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one coordinate {coordinate_id}")
    item = matches[0]
    expected = {
        "model": "474m",
        "batchSequences": 256,
        "gpuCount": 8,
        "rankMicrobatchSequences": 16,
        "gradientAccumulationSteps": 2,
        "learningRate": "2e-3",
    }
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError("474M WD-probe coordinate topology or recipe mismatch")
    wd = str(item["weightDecay"])
    if wd not in {"0.033", "0.3"}:
        raise ValueError("WD probe must be exact WD0.033 or WD0.3")
    expected_source = producer.ALLOWED_ROOT / "dense_474m_dclm1b" / (
        f"bs256_dr_wt_embwd_lr2e-3_wd{wd}"
    )
    expected_output = producer.ALLOWED_ROOT / "dense_474m_dclm3b" / (
        f"bs256_dr_wt_embwd_lr2e-3_wd{wd}"
    )
    if Path(item["sourceOutput"]) != expected_source or Path(item["output"]) != expected_output:
        raise ValueError("WD-probe source/output is not the canonical exact-WD coordinate")
    if check_filesystem:
        producer.validate_pool_lineage(config)
    return config, item


def claim(root: Path, coordinate_id: str) -> None:
    path = root / STATE_NAME / "owner.json"
    job = os.environ.get("BEAKER_JOB_ID", "local")
    experiment = os.environ.get("BEAKER_EXPERIMENT_ID", "local")
    if path.is_file():
        value = json.loads(path.read_text())
        if value.get("coordinate") != coordinate_id or value.get("experiment") != experiment:
            raise RuntimeError(f"output is already claimed by another writer: {root}")
        return
    atomic_json(
        path,
        {"policy": POLICY, "coordinate": coordinate_id, "experiment": experiment, "job": job},
    )


def original_config(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "474m",
        "globalSequences": 256,
        "nprocPerNode": 8,
        "rankMicrobatchSequences": 16,
        "warmupSteps": 96,
        "learningRate": "2e-3",
        "outputRoot": str(Path(item["sourceOutput"]).parent),
        "runSuffix": "pool3b_wd_probe_v1",
    }


def ensure_pool1b_source(item: dict[str, Any]) -> Path:
    source_output = Path(item["sourceOutput"])
    claim(source_output, str(item["id"]))
    source = source_output / f"step{producer.stable_step(1, producer.SOURCE_POOL_TOKENS, 256)}"
    if not source.is_dir():
        original.run_stage(original_config(item), str(item["weightDecay"]), 0, 1)
    item_for_validation = {**item, "sourceCheckpoint": str(source)}
    producer.validate_source_checkpoint(item_for_validation)
    return source


def stage_arguments(
    config: dict[str, Any], item: dict[str, Any], source: Path, epoch: int
) -> list[str]:
    step = producer.stable_step(epoch, producer.TARGET_POOL_TOKENS, 256)
    name = f"{item['id']}-constant-pd-e{epoch}"
    arguments = producer.base_arguments(item, str(config["repeatedManifest"]))
    arguments = producer.common.upsert(
        arguments,
        "--data_loader.restore_data_order_from_state=",
        "--data_loader.restore_data_order_from_state=true",
    )
    return [
        *arguments,
        "--dynamic-repacking",
        f"--save-folder={item['output']}",
        f"--trainer.max_duration={{value: {step}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,pool3b-repeat,"
            f"dense-474m,dr_wt_embwd,bs256,lr2e-3,wd{item['weightDecay']},"
            f"constant-lr,pre-decay,e{epoch},wd-probe-v1]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{step}]",
        f"--trainer.load_path={source}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        "--force_exact_trainer_load_path=true",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def ensure_bridge(config: dict[str, Any], item: dict[str, Any], source: Path) -> Path:
    output = Path(item["output"])
    claim(output, str(item["id"]))
    bridge = output / f"step{producer.stable_step(1, producer.TARGET_POOL_TOKENS, 256)}"
    if not bridge.is_dir():
        unexpected = [path for path in output.glob("step*")]
        if unexpected:
            raise RuntimeError(f"refusing bridge into output with checkpoints: {unexpected}")
        bridge_item = {**item, "sourceCheckpoint": str(source)}
        producer.run_torch(
            f"{item['id']}-fresh-2b-bridge-to-pd-e1",
            producer.bridge_arguments(config, bridge_item),
            output / STATE_NAME / "bridge.log",
        )
    if not bridge.is_dir():
        raise RuntimeError(f"bridge exited without exact Pool-3B E1 checkpoint {bridge}")
    return bridge


def run(config: dict[str, Any], item: dict[str, Any]) -> None:
    source = ensure_pool1b_source(item)
    current = ensure_bridge(config, item, source)
    state_path = Path(item["output"]) / STATE_NAME / "workflow.json"
    results: dict[str, float] = {}
    for epoch in EVALUATION_EPOCHS:
        checkpoint = Path(item["output"]) / (
            f"step{producer.stable_step(epoch, producer.TARGET_POOL_TOKENS, 256)}"
        )
        if not checkpoint.is_dir():
            producer.run_torch(
                f"{item['id']}-constant-pd-e{epoch}",
                stage_arguments(config, item, current, epoch),
                Path(item["output"]) / STATE_NAME / f"producer_e{epoch}.log",
            )
        if not checkpoint.is_dir():
            raise RuntimeError(f"producer exited without exact E{epoch} PD checkpoint")
        result = evaluator.run(config, item, epoch)
        results[str(epoch)] = float(result["validationExact"])
        current = checkpoint
        atomic_json(
            state_path,
            {
                "policy": POLICY,
                "coordinate": item["id"],
                "status": "running" if epoch < EVALUATION_EPOCHS[-1] else "complete",
                "completedEpoch": epoch,
                "evaluationEpochs": list(EVALUATION_EPOCHS),
                "validationExact": results,
            },
        )
    print(
        "DENSE474M_POOL3B_WD_PROBE_COMPLETE json="
        + json.dumps(
            {"policy": POLICY, "coordinate": item["id"], "validationExact": results},
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config, item = load(
        args.manifest, args.coordinate, check_filesystem=not args.validate_only
    )
    if args.validate_only:
        print(f"validated {item['id']}")
        return
    run(config, item)


if __name__ == "__main__":
    main()
