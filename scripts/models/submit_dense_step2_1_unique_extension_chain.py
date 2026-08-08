#!/usr/bin/env python3
"""Submit a 6B/8B/10B/12B/16B unique-data WSD continuation chain."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import subprocess
import sys
from typing import Any


TARGETS = (6, 8, 10, 12, 16)
TOKENS_PER_TARGET = 1_000_000_000
SEQUENCE_LENGTH = 4096
REFERENCE_BATCH_SEQUENCES = 1024
REFERENCE_WARMUP_STEPS = 24
DECAY_FRACTION = 0.1
LEARNING_RATE = "1e-3"
WEIGHT_DECAY = "0.033"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--global-sequences", type=int, choices=(64, 256, 1024), required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    output = [value for value in arguments if not value.startswith(prefix)]
    output.append(replacement)
    return output


def total_steps(target: int, global_sequences: int) -> int:
    return math.ceil(target * TOKENS_PER_TARGET / (global_sequences * SEQUENCE_LENGTH))


def stable_step(target: int, global_sequences: int) -> int:
    end_step = total_steps(target, global_sequences)
    return end_step - round(DECAY_FRACTION * end_step) - 1


def warmup_steps(global_sequences: int) -> int:
    return REFERENCE_WARMUP_STEPS * REFERENCE_BATCH_SEQUENCES // global_sequences


def build_chain(
    base_spec: dict[str, Any],
    *,
    revision: str,
    global_sequences: int,
    resume_checkpoint: str,
    subset_manifest: str,
    pool_manifest: str,
    nproc: int,
    priority: str,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("expected a single-endpoint Python training task")
    training_script = original[1]
    arguments = original[3:]
    warmup = warmup_steps(global_sequences)
    output_root = "/weka/oe-training-default/sewonm/icsl/models"
    commands = [
        "set -euo pipefail",
        shlex.join(["test", "-d", resume_checkpoint]),
        shlex.join(["test", "-f", subset_manifest]),
        shlex.join(["test", "-f", pool_manifest]),
        "python - <<'PY'\n"
        "import json\n"
        f"pool=json.load(open({pool_manifest!r}))\n"
        f"chunk=json.load(open({subset_manifest!r}))\n"
        "assert pool['format']=='dclm-unique-extension-pool-v1'\n"
        "assert pool['logical_pool']['requested_tokens']==1019000000000\n"
        "assert pool['audit']['passed'] is True\n"
        "assert chunk['disjointness_audit']['passed'] is True\n"
        "assert chunk['disjointness_audit']['original_document_overlap']==0\n"
        "assert chunk['disjointness_audit']['validation_test_document_overlap']==0\n"
        "assert chunk['selection']['selected_tokens']>=11500781568\n"
        "print('EXTENSION_PREFLIGHT_OK')\n"
        "PY",
    ]
    previous_output: str | None = None

    for index, target in enumerate(TARGETS):
        name = (
            "dense_1b_step2_1_0802_unique_ext1019b_wsd_"
            f"bs{global_sequences}_t{target}b_lr{LEARNING_RATE}_"
            f"wd{WEIGHT_DECAY}_warmup{warmup}"
        )
        output = f"{output_root}/{name}"
        load_path = resume_checkpoint if index == 0 else (
            f"{previous_output}/step{stable_step(TARGETS[index - 1], global_sequences)}"
        )
        phase_args = [
            value
            for value in arguments
            if not value.startswith(
                (
                    "--load_path=",
                    "--load_trainer_state=",
                    "--trainer.load_path=",
                    "--trainer.load_trainer_state=",
                    "--trainer.load_optim_state=",
                    "--trainer.reset_data_loader_state_on_load_path=",
                )
            )
        ]
        replacements = (
            ("--save-folder=", f"--save-folder={output}"),
            ("--dataset.subset_manifest=", f"--dataset.subset_manifest={subset_manifest}"),
            ("--dataset.mix=", "--dataset.mix=null"),
            (
                "--trainer.max_duration=",
                f"--trainer.max_duration={{value: {target * TOKENS_PER_TARGET}, unit: tokens}}",
            ),
            ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
            (
                "--trainer.callbacks.wandb.tags=",
                "--trainer.callbacks.wandb.tags="
                + "[pretraining, step2-1, 0802, unique-data, dclm-train-only, "
                + "document-disjoint, unique-extension-1019b, data-loader-reset-once, "
                + f"batch-size-tune, wsd, bs{global_sequences}, warmup{warmup}, single-job-chain]",
            ),
            (
                "--data_loader.global_batch_size=",
                f"--data_loader.global_batch_size={global_sequences * SEQUENCE_LENGTH}",
            ),
            (
                "--train_module.scheduler=",
                "--train_module.scheduler="
                + "{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, "
                + f"warmup: {warmup}, decay_fraction: {DECAY_FRACTION}}}",
            ),
            (
                "--trainer.callbacks.checkpointer.fixed_steps=",
                "--trainer.callbacks.checkpointer.fixed_steps="
                + f"[{stable_step(target, global_sequences)}]",
            ),
            (
                "--train_module.optim.weight_decay=",
                f"--train_module.optim.weight_decay={WEIGHT_DECAY}",
            ),
            ("--lr=", f"--lr={LEARNING_RATE}"),
            ("--trainer.load_path=", f"--trainer.load_path={load_path}"),
            ("--trainer.load_trainer_state=", "--trainer.load_trainer_state=true"),
            ("--trainer.load_optim_state=", "--trainer.load_optim_state=true"),
            (
                "--trainer.reset_data_loader_state_on_load_path=",
                "--trainer.reset_data_loader_state_on_load_path="
                + ("true" if index == 0 else "false"),
            ),
        )
        for prefix, replacement in replacements:
            phase_args = upsert(phase_args, prefix, replacement)
        commands.extend(
            (
                shlex.join(["test", "-d", load_path]),
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        f"EXTENSION_CHAIN_START bs={global_sequences} target={target}b "
                        f"source={load_path} "
                        f"reset_data_loader={str(index == 0).lower()}",
                    ]
                ),
                shlex.join(["python", training_script, name, "--dry-run", *phase_args]),
                shlex.join(
                    ["torchrun", f"--nproc-per-node={nproc}", training_script, name, *phase_args]
                ),
                shlex.join(
                    ["test", "-d", f"{output}/step{stable_step(target, global_sequences)}"]
                ),
                shlex.join(
                    ["echo", f"EXTENSION_CHAIN_FINISH bs={global_sequences} target={target}b"]
                ),
            )
        )
        previous_output = output

    commands.append(
        shlex.join(
            [
                "echo",
                f"EXTENSION_CHAIN_COMPLETE bs={global_sequences} "
                f"targets={','.join(map(str, TARGETS))}",
            ]
        )
    )
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["envVars"] = [
        item for item in task.get("envVars", []) if item.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    for item in task["envVars"]:
        if item.get("name") == "GIT_REF":
            item["value"] = revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": revision})
    task["resources"] = {"gpuCount": nproc, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    spec = build_chain(
        get_base_spec(args.base_experiment),
        revision=args.revision,
        global_sequences=args.global_sequences,
        resume_checkpoint=args.resume_checkpoint,
        subset_manifest=args.subset_manifest,
        pool_manifest=args.pool_manifest,
        nproc=args.nproc,
        priority=args.priority,
    )
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            args.name,
            "--workspace",
            args.workspace,
            "--priority",
            args.priority,
        ],
        check=True,
        input=json.dumps(spec),
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
