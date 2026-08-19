#!/usr/bin/env python3
"""Run one retry-safe BS32 small-dense saturation chain.

Every WSD endpoint is evaluated in sequence.  The following endpoint resumes the
preceding endpoint's exact pre-decay checkpoint while every endpoint writes to
the same canonical save folder.  Atomic result and decision files make Beaker
task retries idempotent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([0-9]+(?:\.[0-9]+)?)")
VALIDATION_LOSS = re.compile(
    r"\bdclm-validation-0802/CE loss=([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)
WANDB_VALIDATION_LOSS = re.compile(
    r"wandb:\s+eval/heldout/dclm-validation-0802/CE loss\s+"
    r"([0-9]+(?:\.[0-9]+)?)"
)
WANDB_RUN = re.compile(r"https://wandb\.ai/[^\s]+/runs/([a-zA-Z0-9_-]+)")
ACCURACY = re.compile(r"\s([a-z0-9_]+) \((?:length-normalized )?accuracy\)=([0-9.]+)")
BPB = re.compile(r"\s([a-z0-9_]+) \(BPB\)=([0-9.]+)")
REPORT_TASKS = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
AVERAGE_TASKS = tuple(task for task in REPORT_TASKS if task != "boolq")
FORBIDDEN_ARGUMENTS = (
    "--dynamic-repacking",
    "--batch-shuffling",
    "--fixed-data-order",
    "--no-data-shuffle",
    "--decay-embeddings",
    "--mlp-weight-decay",
)


def total_step(epoch: int, global_sequences: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / (global_sequences * SEQUENCE_LENGTH))


def stable_step(epoch: int, global_sequences: int) -> int:
    end = total_step(epoch, global_sequences)
    return end - round(DECAY_FRACTION * end) - 1


def normalize_task(name: str) -> str:
    for prefix in ("csqa", "openbookqa", "socialiqa"):
        if name.startswith(prefix):
            return prefix
    return name


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    found = False
    output: list[str] = []
    for argument in arguments:
        if argument.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(argument)
    if not found:
        output.append(replacement)
    return output


def atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents)
    os.replace(temporary, path)


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model",
        "script",
        "output",
        "baseArguments",
        "globalSequences",
        "nprocPerNode",
        "warmupSteps",
        "learningRate",
        "weightDecay",
        "previousEpoch",
        "firstEpoch",
        "epochIncrement",
        "previousValidation",
        "initialSourceCheckpoint",
        "runSuffix",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    if int(config["globalSequences"]) != 32:
        raise ValueError("this persistent launcher is scoped to BS32")
    if int(config["nprocPerNode"]) != 2:
        raise ValueError("BS32 must use two GPUs")
    if int(config["firstEpoch"]) - int(config["previousEpoch"]) != int(config["epochIncrement"]):
        raise ValueError("first epoch must be exactly one increment past the frontier")
    if not str(config["initialSourceCheckpoint"]).endswith(
        f"/step{stable_step(int(config['previousEpoch']), 32)}"
    ):
        raise ValueError("initial source is not the exact frontier pre-decay checkpoint")
    arguments = [str(argument) for argument in config["baseArguments"]]
    for forbidden in FORBIDDEN_ARGUMENTS:
        if any(
            argument == forbidden or argument.startswith(forbidden + "=") for argument in arguments
        ):
            raise ValueError(f"default setup forbids {forbidden}")
    if "--model.tie_embeddings=false" not in arguments:
        raise ValueError("the manifest must explicitly keep weight tying disabled")
    embedding_policy = config.get("embeddingWeightDecay")
    if embedding_policy != "zero":
        raise ValueError("the default embedding weight-decay policy must be zero")


def emit_result(config: dict[str, Any], epoch: int, result: dict[str, Any]) -> None:
    payload = json.dumps(result, separators=(",", ":"), sort_keys=True)
    print(
        f"SMALL_SATURATION_STAGE_RESULT model={config['model']} epoch={epoch} json={payload}",
        flush=True,
    )


def stream_command(command: list[str], log_file: TextIO) -> None:
    print("SMALL_SATURATION_COMMAND " + " ".join(command[:4]), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_stage_result(log_path: Path, epoch: int) -> dict[str, Any]:
    clean = ANSI.sub("", log_path.read_text())
    train_values = TRAIN_LOSS.findall(clean)
    validation_values = WANDB_VALIDATION_LOSS.findall(clean) or VALIDATION_LOSS.findall(clean)
    if not train_values or not validation_values:
        raise RuntimeError(f"E{epoch} completed without train and held-out validation CE")

    accuracy: dict[str, float] = {}
    bpb: dict[str, float] = {}
    for task, value in ACCURACY.findall(clean):
        accuracy[normalize_task(task)] = 100 * float(value)
    for task, value in BPB.findall(clean):
        bpb[normalize_task(task)] = float(value)
    missing = [task for task in REPORT_TASKS if task not in accuracy or task not in bpb]
    if missing:
        raise RuntimeError(f"E{epoch} completed without downstream metrics for {missing}")

    train = float(train_values[-1])
    validation = float(validation_values[-1])
    result: dict[str, Any] = {
        "epoch": epoch,
        "status": "complete",
        "train": train,
        "validation": round(validation, 3),
        "validationExact": validation,
        "gap": round(validation - train, 6),
        "downstream": accuracy,
        "downstreamBpb": bpb,
        "acc": accuracy["hellaswag"],
        "bpb": bpb["hellaswag"],
        "avg8Accuracy": sum(accuracy[task] for task in AVERAGE_TASKS) / len(AVERAGE_TASKS),
        "avg8Bpb": sum(bpb[task] for task in AVERAGE_TASKS) / len(AVERAGE_TASKS),
        "c4": round(validation, 3),
    }
    wandb_values = WANDB_RUN.findall(clean)
    if wandb_values:
        result["wandb"] = wandb_values[-1]
    return result


def recovery_checkpoint(
    config: dict[str, Any], previous_epoch: int, epoch: int, output: Path
) -> Path:
    endpoint = output / f"step{total_step(epoch, int(config['globalSequences']))}"
    if endpoint.is_dir():
        print(f"SMALL_SATURATION_EVAL_RECOVERY epoch={epoch} checkpoint={endpoint}", flush=True)
        return endpoint

    candidates = [
        output / f"step{stable_step(candidate_epoch, int(config['globalSequences']))}"
        for candidate_epoch in range(previous_epoch + 1, epoch + 1)
    ]
    if previous_epoch == int(config["previousEpoch"]):
        candidates.insert(0, Path(str(config["initialSourceCheckpoint"])))
    else:
        candidates.insert(
            0,
            output / f"step{stable_step(previous_epoch, int(config['globalSequences']))}",
        )
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if not existing:
        raise FileNotFoundError(
            f"no exact pre-decay recovery checkpoint exists for E{epoch}: {candidates}"
        )
    selected = existing[-1]
    print(f"SMALL_SATURATION_RESUME epoch={epoch} checkpoint={selected}", flush=True)
    return selected


def stage_arguments(
    config: dict[str, Any], previous_epoch: int, epoch: int, checkpoint: Path
) -> tuple[str, list[str]]:
    arguments = [str(argument) for argument in config["baseArguments"]]
    output = str(config["output"])
    name = (
        f"dense_{config['model']}_step1_0802_repeated_dclm1b_bs32_"
        f"e{epoch}_lr{config['learningRate']}_wd{config['weightDecay']}_"
        f"warmup{config['warmupSteps']}_{config['runSuffix']}"
    )
    retained = [
        stable_step(candidate_epoch, int(config["globalSequences"]))
        for candidate_epoch in range(previous_epoch + 1, epoch + 1)
    ]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        (
            "--trainer.max_duration=",
            f"--trainer.max_duration={{value: {epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            (
                "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,"
                f"dense-{config['model']},repeated-data,dclm-train-only,batch-size-tune,"
                f"wsd,bs32,e{epoch},persistent-saturation-chain]"
            ),
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            "--trainer.callbacks.checkpointer.fixed_steps="
            + json.dumps(retained, separators=(",", ":")),
        ),
        (
            "--train_module.scheduler=",
            (
                "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
                f"units: steps, warmup: {config['warmupSteps']}, "
                f"decay_fraction: {DECAY_FRACTION}}}"
            ),
        ),
        ("--force_exact_trainer_load_path=", "--force_exact_trainer_load_path=true"),
        ("--trainer.load_path=", f"--trainer.load_path={checkpoint}"),
        ("--trainer.load_trainer_state=", "--trainer.load_trainer_state=true"),
        ("--trainer.load_optim_state=", "--trainer.load_optim_state=true"),
        (
            "--trainer.reset_data_loader_state_on_load_path=",
            "--trainer.reset_data_loader_state_on_load_path=false",
        ),
        (
            "--train_module.validate_optimizer_hyperparameters_on_load=",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        ),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    return name, arguments


def run(config: dict[str, Any]) -> None:
    validate_config(config)
    output = Path(str(config["output"]))
    output.mkdir(parents=True, exist_ok=True)
    previous_epoch = int(config["previousEpoch"])
    previous_validation = float(config["previousValidation"])
    epoch = int(config["firstEpoch"])
    increment = int(config["epochIncrement"])

    while True:
        result_path = output / f".bs32_saturation_e{epoch}.result.json"
        decision_path = output / f".bs32_saturation_e{epoch}.decision"
        if result_path.is_file() and decision_path.is_file():
            result = json.loads(result_path.read_text())
            emit_result(config, epoch, result)
            action = decision_path.read_text().strip()
            print(
                f"SMALL_SATURATION_STAGE_REUSED model={config['model']} epoch={epoch} "
                f"validation={result['validationExact']} action={action}",
                flush=True,
            )
            if action == "stop":
                print(
                    f"SMALL_SATURATION_SATURATED model={config['model']} epoch={epoch}",
                    flush=True,
                )
                return
            previous_validation = float(result["validationExact"])
            previous_epoch = epoch
            epoch += increment
            continue

        checkpoint = recovery_checkpoint(config, previous_epoch, epoch, output)
        name, arguments = stage_arguments(config, previous_epoch, epoch, checkpoint)
        print(
            f"SMALL_SATURATION_STAGE_START model={config['model']} epoch={epoch} "
            f"previous_epoch={previous_epoch} previous_validation={previous_validation} "
            f"output={output}",
            flush=True,
        )
        dry_run = ["python", str(config["script"]), name, "--dry-run", *arguments]
        subprocess.run(dry_run, check=True)

        log_path = output / f".bs32_saturation_e{epoch}.log"
        with log_path.open("a") as log_file:
            stream_command(
                [
                    "torchrun",
                    f"--nproc-per-node={config['nprocPerNode']}",
                    str(config["script"]),
                    name,
                    *arguments,
                ],
                log_file,
            )
        result = parse_stage_result(log_path, epoch)
        result.update(
            {
                "lr": str(config["learningRate"]),
                "wd": str(config["weightDecay"]),
                "output": str(output),
                "sourceCheckpoint": str(checkpoint),
                "preDecayCheckpoint": str(
                    output / f"step{stable_step(epoch, int(config['globalSequences']))}"
                ),
                "endpointCheckpoint": str(
                    output / f"step{total_step(epoch, int(config['globalSequences']))}"
                ),
                "sequential": True,
            }
        )
        action = "continue" if float(result["validationExact"]) < previous_validation else "stop"
        result["saturationDecision"] = action
        result["previousValidationExact"] = previous_validation
        atomic_write(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
        atomic_write(decision_path, action + "\n")
        emit_result(config, epoch, result)
        print(
            f"SMALL_SATURATION_STAGE_COMPLETE model={config['model']} epoch={epoch} "
            f"validation={result['validationExact']} previous={previous_validation} "
            f"action={action}",
            flush=True,
        )
        if action == "stop":
            print(
                f"SMALL_SATURATION_SATURATED model={config['model']} epoch={epoch}",
                flush=True,
            )
            return
        previous_validation = float(result["validationExact"])
        previous_epoch = epoch
        epoch += increment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    validate_config(config)
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    run(config)


if __name__ == "__main__":
    main()
