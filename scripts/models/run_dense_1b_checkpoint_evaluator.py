#!/usr/bin/env python3
"""Independently WSD-decay and evaluate the retained 1B E8/E16 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_dense_constant_checkpoint_producer as producer
import run_small_dense_saturation_chain as common

POLICY = "dense_1b_checkpoint_evaluator_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(manifest_path: Path, evaluator_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = producer.load_manifest(manifest_path)
    matches = [
        item
        for item in manifest.get("evaluatorCoordinates", [])
        if item.get("id") == evaluator_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one evaluator {evaluator_id}")
    evaluator = matches[0]
    item = producer.coordinate(manifest, str(evaluator["producerId"]))
    epochs = [int(epoch) for epoch in evaluator.get("epochs", [])]
    if item["model"] != "1b" or epochs != [8, 16]:
        raise ValueError("only the two 1B E8/E16 evaluators are authorized")
    producer.validate_coordinate(item, int(manifest["maxEpoch"]), check_source=False)
    return item, evaluator


def state_dir(item: dict[str, Any]) -> Path:
    return producer.state_dir(item) / "evaluator"


def source_checkpoint(item: dict[str, Any], epoch: int) -> Path:
    step = producer.stable_step(
        epoch, int(item["poolTokens"]), int(item["batchSequences"])
    )
    return Path(str(item["output"])) / f"step{step}"


def validate_source(item: dict[str, Any], epoch: int) -> Path:
    source = source_checkpoint(item, epoch)
    source_item = {**item, "sourceEpoch": epoch, "sourceCheckpoint": str(source)}
    producer.validate_source_checkpoint(source_item)
    return source


def base_arguments(item: dict[str, Any], *, heldout: str) -> list[str]:
    arguments = producer.base_arguments(item)
    return common.upsert(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )


def run_torch(name: str, arguments: list[str], log_path: Path) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            ["torchrun", "--nproc-per-node=8", TRAINING_SCRIPT, name, *arguments], log_file
        )


def parse_result(
    log_path: Path, item: dict[str, Any], epoch: int, checkpoint: Path, source: Path
) -> dict[str, Any]:
    result = dense1b.parse_validation(log_path, epoch, "post_decay", checkpoint)
    result.update(
        {
            "policy": POLICY,
            "producerPolicy": producer.POLICY,
            "lr": str(item["learningRate"]),
            "wd": str(item["weightDecay"]),
            "variant": "DR+WT+EmbedWD",
            "dynamicRepacking": True,
            "weightTying": True,
            "decayEmbeddings": True,
            "sourcePreDecayCheckpoint": str(source),
            "source": "independent_wsd_decay_and_heldout_eval",
        }
    )
    return result


def evaluation_arguments(
    item: dict[str, Any], checkpoint: Path, output: Path, name: str
) -> list[str]:
    heldout = dense1b.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    return [
        *base_arguments(item, heldout=heldout),
        f"--save-folder={output}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-1b,checkpoint-evaluator,post-decay]",
        "--trainer.max_duration={value: 1000000000000, unit: steps}",
        "--trainer.callbacks.checkpointer.enabled=false",
        f"--load_path={checkpoint}",
        "--load_trainer_state=false",
    ]


def postdecay_arguments(
    item: dict[str, Any], epoch: int, source: Path, output: Path, name: str
) -> list[str]:
    endpoint = producer.total_step(
        epoch, int(item["poolTokens"]), int(item["batchSequences"])
    )
    return [
        *base_arguments(item, heldout=dense1b.HELDOUT_EVALUATOR),
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {endpoint}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dense-1b,"
            f"checkpoint-evaluator,post-decay,bs{item['batchSequences']},e{epoch}]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{endpoint}]",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--data_loader.restore_data_order_from_state=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
            f"units: steps, warmup: {24576 // int(item['batchSequences'])}, "
            f"decay_fraction: {producer.DECAY_FRACTION}}}"
        ),
        "--dynamic-repacking",
        "--force_exact_trainer_load_path=true",
        f"--trainer.load_path={source}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def run_epoch(item: dict[str, Any], epoch: int) -> dict[str, Any]:
    result_path = state_dir(item) / "post_decay" / f"e{epoch}.result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    source = validate_source(item, epoch)
    output = state_dir(item) / "post_decay_runs" / f"e{epoch}"
    endpoint_step = producer.total_step(
        epoch, int(item["poolTokens"]), int(item["batchSequences"])
    )
    endpoint = output / f"step{endpoint_step}"
    name = f"{item['id']}-postdecay-e{epoch}-v1"
    if endpoint.is_dir():
        eval_name = f"{name}-recovered-eval"
        log_path = state_dir(item) / "logs" / f"postdecay_e{epoch}_recovered_eval.log"
        run_torch(
            eval_name,
            evaluation_arguments(item, endpoint, output / "eval", eval_name),
            log_path,
        )
    else:
        log_path = state_dir(item) / "logs" / f"postdecay_e{epoch}.log"
        print(
            f"DENSE1B_CHECKPOINT_EVALUATOR_START id={item['id']} epoch={epoch} "
            f"source={source} output={output}",
            flush=True,
        )
        run_torch(name, postdecay_arguments(item, epoch, source, output, name), log_path)
    if not endpoint.is_dir():
        raise RuntimeError(f"E{epoch} decay exited without endpoint {endpoint}")
    result = parse_result(log_path, item, epoch, endpoint, source)
    atomic_json(result_path, result)
    print(
        f"DENSE1B_CHECKPOINT_EVALUATOR_RESULT id={item['id']} epoch={epoch} "
        f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    return result


def run(item: dict[str, Any], epochs: list[int]) -> None:
    results = {epoch: run_epoch(item, epoch) for epoch in epochs}
    previous, current = results[epochs[-2]], results[epochs[-1]]
    saturated = Decimal(str(current["validationExact"])) >= Decimal(
        str(previous["validationExact"])
    )
    decision = {
        "policy": POLICY,
        "status": "saturated" if saturated else "improving",
        "comparisonGroup": "post_decay_only",
        "criterion": "strict_non_improvement",
        "epochs": epochs,
        "validationExact": {
            str(epoch): float(results[epoch]["validationExact"]) for epoch in epochs
        },
        "producerCancellationRequested": False,
    }
    atomic_json(state_dir(item) / "decision.json", decision)
    print(
        f"DENSE1B_CHECKPOINT_EVALUATOR_COMPLETE id={item['id']} status={decision['status']} "
        f"json={json.dumps(decision, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    item, evaluator = load(args.manifest, args.evaluator)
    if args.validate_only:
        print(f"validated evaluator {args.evaluator}")
        return
    run(item, [int(epoch) for epoch in evaluator["epochs"]])


if __name__ == "__main__":
    main()
