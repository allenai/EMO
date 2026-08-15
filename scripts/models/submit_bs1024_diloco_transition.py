#!/usr/bin/env python3
"""Submit one guarded BS1024 conventional-to-DiLoCo transition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REVISION = "15d494fee454eda89884f6f0dc4535a0c0ad556d"
BRANCH = "sewonm/icsl"
ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
GLOBAL_BATCH_TOKENS = 1024 * 4096
RANK_MICROBATCH_TOKENS = 4 * 4096
FRONTIERS = {2: (428, 477), 4: (858, 954), 8: (1716, 1908)}
TEMPLATES = {
    64: "01KZZ4BG861VPER8SFZXZK29AT",
    256: "01KZZ4BTYFP6T7XHVJZK754DVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", type=int, choices=(4, 32), required=True)
    parser.add_argument("--simulated-sequences", type=int, choices=(64, 256), required=True)
    parser.add_argument("--wd", choices=("0.333", "1.0"), required=True)
    parser.add_argument("--init-epoch", type=int, choices=(2, 4), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.h == 32 and (args.simulated_sequences, args.init_epoch, args.wd) != (256, 4, "1.0"):
        parser.error("the authorized H32 addition is only simBS256, E4 init, WD1.0")
    if args.h == 4 and args.simulated_sequences != 64:
        parser.error("the authorized H4 additions are simBS64 only")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name")
    return args


def source_checkpoint(args: argparse.Namespace) -> str:
    step = FRONTIERS[args.init_epoch][0]
    return f"{ROOT}/bs1024_dr_lr1e-3_wd{args.wd}/step{step}"


def output_path(args: argparse.Namespace) -> str:
    return (
        f"{ROOT}/bs1024_dr_diloco_h{args.h}_vrecal_simbs{args.simulated_sequences}_"
        f"init=bs1024e{args.init_epoch}_lr1e-3_wd{args.wd}"
    )


def source_spec(simulated_sequences: int) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", TEMPLATES[simulated_sequences], "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def extract_training(spec: dict[str, Any]) -> tuple[str, list[str]]:
    for line in spec["tasks"][0]["arguments"][2].splitlines():
        if not line.startswith("torchrun "):
            continue
        parts = shlex.split(line)
        indices = [i for i, value in enumerate(parts) if value.endswith("olmo2-1B.py")]
        if len(indices) == 1:
            index = indices[0]
            return parts[index], parts[index + 2 :]
    raise RuntimeError("could not extract source training command")


def upsert(values: list[str], prefix: str, replacement: str) -> list[str]:
    output: list[str] = []
    found = False
    for value in values:
        if value.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(value)
    if not found:
        output.append(replacement)
    return output


def outer_steps(source_step: int, pre_decay: int, endpoint: int, horizon: int) -> list[int]:
    steps: list[int] = []
    cursor = source_step
    for boundary in (pre_decay, endpoint):
        while cursor + horizon < boundary:
            cursor += horizon
            steps.append(cursor)
        steps.append(boundary)
        cursor = boundary
    return steps


def training_arguments(
    source: list[str], args: argparse.Namespace, output: str, pre_decay: int, endpoint: int
) -> list[str]:
    excluded = (
        "--trainer.load_path=",
        "--trainer.load_trainer_state=",
        "--trainer.load_optim_state=",
        "--trainer.prefer_explicit_load_path=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--train_module.batch_simulation.",
        "--train_module.validate_optimizer_hyperparameters_on_load=",
        "--trainer.callbacks.checkpointer.save_interval=",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
    )
    values = [value for value in source if not value.startswith(excluded)]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {args.init_epoch * 2 * 1_000_000_000}, unit: tokens}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={Path(output).name}_e{args.init_epoch * 2}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            f"[pretraining,step1,local-update,bs1024,simbs{args.simulated_sequences},diloco,"
            f"h{args.h},dynamic-repacking,e{args.init_epoch * 2},lr1e-3,wd{args.wd},"
            f"conventional-e{args.init_epoch}-init]",
        ),
        ("--trainer.callbacks.checkpointer.fixed_steps=", f"--trainer.callbacks.checkpointer.fixed_steps=[{pre_decay}]"),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={GLOBAL_BATCH_TOKENS}"),
        ("--data_loader.restore_data_order_from_state=", "--data_loader.restore_data_order_from_state=true"),
        ("--data_loader.ignore_fingerprint_mismatch=", "--data_loader.ignore_fingerprint_mismatch=false"),
        ("--train_module.rank_microbatch_size=", f"--train_module.rank_microbatch_size={RANK_MICROBATCH_TOKENS}"),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 24, decay_fraction: 0.1}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={args.wd}"),
        ("--lr=", "--lr=1e-3"),
    )
    for prefix, replacement in replacements:
        values = upsert(values, prefix, replacement)
    source_step = FRONTIERS[args.init_epoch][0]
    values.extend(
        [
            f"--trainer.load_path={source_checkpoint(args)}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.prefer_explicit_load_path=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            f"--train_module.batch_simulation.global_batch_size={GLOBAL_BATCH_TOKENS}",
            f"--train_module.batch_simulation.simulated_batch_size={args.simulated_sequences * 4096}",
            "--train_module.batch_simulation.seed=12536",
            "--train_module.batch_simulation.recalibrate_second_moment_on_start=true",
            "--train_module.batch_simulation.method=diloco",
            f"--train_module.batch_simulation.diloco_inner_steps={args.h}",
            "--train_module.batch_simulation.diloco_outer_lr=0.7",
            "--train_module.batch_simulation.diloco_outer_momentum=0.9",
            "--train_module.batch_simulation.diloco_outer_steps="
            + json.dumps(outer_steps(source_step, pre_decay, endpoint, args.h), separators=(",", ":")),
        ]
    )
    return values


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    task.setdefault("envVars", [])[:] = [env for env in task.get("envVars", []) if env.get("name") != name]
    task["envVars"].append({"name": name, "value": value})


def build_spec(
    spec: dict[str, Any], script: str, train_args: list[str], args: argparse.Namespace, output: str
) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    task = copy.deepcopy(spec["tasks"][0])
    spec["tasks"] = [task]
    target_epoch = args.init_epoch * 2
    pre_decay, endpoint = FRONTIERS[target_epoch]
    nodes = 2 if args.simulated_sequences == 64 else 1
    preflight = [
        "set -euo pipefail",
        shlex.join(["test", "-d", source_checkpoint(args)]),
        shlex.join(["test", "-f", TRAIN_MANIFEST]),
        shlex.join(["test", "!", "-e", output]),
        f'test "$(git rev-parse HEAD)" = "{REVISION}"',
        shlex.join(["python", script, Path(output).name, "--dry-run", *train_args]),
        "python -m pytest -q src/test/train/train_module/transformer/batch_simulation_test.py "
        "-k 'diloco_checkpoint_preserves_replica_optimizer_state or diloco_does_not_expose_raw_replica_checkpointing'",
    ]
    launch_prefix = (
        'torchrun --nnodes="$BEAKER_REPLICA_COUNT:$BEAKER_REPLICA_COUNT" '
        '--nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" --rdzv-id="$GANTRY_RDZV_ID" '
        '--rdzv-backend=static --rdzv-endpoint="$BEAKER_LEADER_REPLICA_HOSTNAME:$GANTRY_RDZV_PORT" '
        '--node-rank="$BEAKER_REPLICA_RANK" --rdzv-conf="read_timeout=420" '
        if nodes == 2
        else 'torchrun --nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
    )
    launch = launch_prefix + shlex.join([script, Path(output).name, *train_args])
    postflight = [
        shlex.join(["test", "-d", f"{output}/step{pre_decay}"]),
        shlex.join(["test", "-d", f"{output}/step{endpoint}"]),
    ]
    task["arguments"] = [
        "bash",
        "-lc",
        "set -euo pipefail\n"
        'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then\n'
        + "\n".join(preflight)
        + "\nfi\n"
        + launch
        + "\n"
        + 'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then\n'
        + "\n".join(postflight)
        + "\nfi",
    ]
    blocked = {"GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in blocked
        and not (str(env.get("name", "")).startswith("BEAKER_") and env.get("name") != "BEAKER_TOKEN")
    ]
    set_env(task, "GIT_REF", REVISION)
    set_env(task, "GIT_BRANCH", BRANCH)
    set_env(task, "NUM_NODES", str(nodes))
    if nodes == 2:
        digest = hashlib.sha256(output.encode()).hexdigest()
        set_env(task, "GANTRY_RDZV_ID", digest[:12])
        set_env(task, "GANTRY_RDZV_PORT", str(29000 + int(digest[:8], 16) % 1000))
        task.update(
            replicas=2,
            leaderSelection=True,
            hostNetworking=True,
            propagateFailure=True,
            propagatePreemption=True,
            synchronizedStartTimeout="90m",
        )
    else:
        for field in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
            task.pop(field, None)
        task["hostNetworking"] = False
        task["propagateFailure"] = False
        task["propagatePreemption"] = False
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    target_epoch = args.init_epoch * 2
    pre_decay, endpoint = FRONTIERS[target_epoch]
    output = output_path(args)
    spec = source_spec(args.simulated_sequences)
    script, source = extract_training(spec)
    train_args = training_arguments(source, args, output, pre_decay, endpoint)
    spec = build_spec(spec, script, train_args, args, output)
    shell = spec["tasks"][0]["arguments"][2]
    required = [
        source_checkpoint(args),
        output,
        f"diloco_inner_steps={args.h}",
        "recalibrate_second_moment_on_start=true",
        f"weight_decay={args.wd}",
        f"step{pre_decay}",
        f"step{endpoint}",
        REVISION,
    ]
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"built spec missing guards: {missing}")
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
        input=json.dumps(spec),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
