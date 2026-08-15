#!/usr/bin/env python3
"""Submit one guarded BS1024 local-update frontier from an exact pre-decay checkpoint."""

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

REVISION = "74b1bf73903ecf5db09c83ed42e36ed449269f90"
BRANCH = "sewonm/icsl"
GLOBAL_SEQUENCES = 1024
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_TOKENS = GLOBAL_SEQUENCES * SEQUENCE_LENGTH
RANK_MICROBATCH_TOKENS = 4 * SEQUENCE_LENGTH
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
FRONTIERS = {
    2: (428, 477),
    4: (858, 954),
    8: (1716, 1908),
    12: (2575, 2862),
    16: (3432, 3815),
    20: (4291, 4769),
    24: (5150, 5723),
}
TEMPLATES = {
    ("local_sgd", 64): "01KZZ4AQYSS6WDPCRV23853FW4",
    ("local_sgd", 256): "01KZZ4B4NAFV9YAKF1MJEKMWY4",
    ("diloco", 64): "01KZZ4BG861VPER8SFZXZK29AT",
    ("diloco", 256): "01KZZ4BTYFP6T7XHVJZK754DVG",
    ("sequential_replay", 64): "01KZZ4AQYSS6WDPCRV23853FW4",
    ("sequential_replay", 256): "01KZZ4B4NAFV9YAKF1MJEKMWY4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("local_sgd", "diloco", "sequential_replay"), required=True
    )
    parser.add_argument("--simulated-sequences", type=int, choices=(64, 256), required=True)
    parser.add_argument("--sync-interval", type=int, choices=(4, 32), default=32)
    parser.add_argument("--diloco-outer-lr", type=float, default=0.7)
    parser.add_argument("--diloco-outer-momentum", type=float, default=0.9)
    parser.add_argument("--wd", choices=("0.333", "1.0"), required=True)
    parser.add_argument("--start-epoch", type=int, choices=tuple(FRONTIERS), required=True)
    parser.add_argument("--target-epoch", type=int, choices=tuple(FRONTIERS), required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--conventional-transition",
        action="store_true",
        help="Start a new local-update trajectory from a conventional checkpoint and run one-time Adam-v recalibration.",
    )
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    sequence = (2, 4, 8, 12, 16, 20, 24)
    if sequence.index(args.target_epoch) != sequence.index(args.start_epoch) + 1:
        parser.error("target epoch must be the next authorized frontier")
    expected_source_step = FRONTIERS[args.start_epoch][0]
    if not args.source_checkpoint.endswith(f"/step{expected_source_step}"):
        parser.error(f"source must be exact pre-decay step{expected_source_step}")
    conventional_source = (
        f"/bs1024_dr_lr1e-3_wd{args.wd}/step{expected_source_step}"
    )
    if args.start_epoch > 2 and args.source_checkpoint.endswith(conventional_source) and not args.conventional_transition:
        parser.error("a conventional source after E2 requires --conventional-transition")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name")
    if args.conventional_transition:
        expected = conventional_source
        if not args.source_checkpoint.endswith(expected):
            parser.error(f"conventional transition source must end with {expected}")
        if f"init=bs1024e{args.start_epoch}_lr1e-3_wd{args.wd}" not in args.output:
            parser.error("conventional transition output must encode its exact initialization epoch")
    return args


def source_spec(method: str, simulated_sequences: int) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", TEMPLATES[(method, simulated_sequences)], "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def extract_training(spec: dict[str, Any]) -> tuple[str, list[str]]:
    task = spec["tasks"][0]
    for line in task["arguments"][2].splitlines():
        if not line.startswith("torchrun "):
            continue
        parts = shlex.split(line)
        indices = [i for i, value in enumerate(parts) if value.endswith("olmo2-1B.py")]
        if len(indices) == 1:
            index = indices[0]
            return parts[index], parts[index + 2 :]
    raise RuntimeError("could not extract source training command")


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    output: list[str] = []
    found = False
    for value in arguments:
        if value.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(value)
    if not found:
        output.append(replacement)
    return output


def outer_steps(source_step: int, pre_decay: int, endpoint: int, inner_steps: int) -> list[int]:
    steps: list[int] = []
    cursor = source_step
    for boundary in (pre_decay, endpoint):
        while cursor + inner_steps < boundary:
            cursor += inner_steps
            steps.append(cursor)
        steps.append(boundary)
        cursor = boundary
    return steps


def training_arguments(
    source: list[str], args: argparse.Namespace, pre_decay: int, endpoint: int
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
        ("--save-folder=", f"--save-folder={args.output}"),
        (
            "--trainer.max_duration=",
            f"--trainer.max_duration={{value: {args.target_epoch * 1_000_000_000}, unit: tokens}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={Path(args.output).name}_e{args.target_epoch}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            f"[pretraining,step1,local-update,bs1024,simbs{args.simulated_sequences},"
            f"{args.method},h{args.sync_interval},dynamic-repacking,e{args.target_epoch},lr1e-3,wd{args.wd}]",
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
    values.extend(
        [
            f"--trainer.load_path={args.source_checkpoint}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.prefer_explicit_load_path=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            f"--train_module.batch_simulation.global_batch_size={GLOBAL_BATCH_TOKENS}",
            f"--train_module.batch_simulation.simulated_batch_size={args.simulated_sequences * SEQUENCE_LENGTH}",
            "--train_module.batch_simulation.seed=12536",
            "--train_module.batch_simulation.recalibrate_second_moment_on_start="
            + ("true" if args.start_epoch == 2 or args.conventional_transition else "false"),
        ]
    )
    if args.method == "local_sgd":
        values.extend(
            [
                "--train_module.batch_simulation.method=local_sgd",
                f"--train_module.batch_simulation.local_sgd_sync_interval={args.sync_interval}",
            ]
        )
    elif args.method == "diloco":
        values.extend(
            [
                "--train_module.batch_simulation.method=diloco",
                f"--train_module.batch_simulation.diloco_inner_steps={args.sync_interval}",
                f"--train_module.batch_simulation.diloco_outer_lr={args.diloco_outer_lr:g}",
                f"--train_module.batch_simulation.diloco_outer_momentum={args.diloco_outer_momentum:g}",
                "--train_module.batch_simulation.diloco_outer_steps="
                + json.dumps(
                    outer_steps(
                        FRONTIERS[args.start_epoch][0],
                        pre_decay,
                        endpoint,
                        args.sync_interval,
                    ),
                    separators=(",", ":"),
                ),
            ]
        )
    else:
        values.extend(
            [
                "--train_module.batch_simulation.method=sequential_replay",
                "--train_module.batch_simulation.sequential_replay_microbatch_gradients=false",
            ]
        )
    return values


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    task.setdefault("envVars", [])[:] = [env for env in task.get("envVars", []) if env.get("name") != name]
    task["envVars"].append({"name": name, "value": value})


def build_spec(spec: dict[str, Any], script: str, train_args: list[str], args: argparse.Namespace) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    task = copy.deepcopy(spec["tasks"][0])
    spec["tasks"] = [task]
    pre_decay, endpoint = FRONTIERS[args.target_epoch]
    nodes = 2 if args.simulated_sequences == 64 else 1
    source_step = FRONTIERS[args.start_epoch][0]
    preflight = [
        "set -euo pipefail",
        shlex.join(["test", "-d", args.source_checkpoint]),
        shlex.join(["test", "-f", TRAIN_MANIFEST]),
        shlex.join(["test", "!", "-e", f"{args.output}/step{pre_decay}"]),
        shlex.join(["test", "!", "-e", f"{args.output}/step{endpoint}"]),
        f'test "$(git rev-parse HEAD)" = "{REVISION}"',
        shlex.join(["python", script, Path(args.output).name, "--dry-run", *train_args]),
    ]
    if args.method == "diloco":
        preflight.append(
            "python -m pytest -q src/test/train/train_module/transformer/batch_simulation_test.py "
            "-k 'diloco_checkpoint_preserves_replica_optimizer_state or "
            "diloco_does_not_expose_raw_replica_checkpointing'"
        )
    elif args.method == "sequential_replay":
        preflight.append(
            "python -m pytest -q src/test/train/train_module/transformer/batch_simulation_test.py "
            "-k 'sequential_replay'"
        )
    if args.start_epoch == 2 or args.conventional_transition:
        preflight.insert(3, shlex.join(["test", "!", "-e", args.output]))
    launch_prefix = (
        'torchrun --nnodes="$BEAKER_REPLICA_COUNT:$BEAKER_REPLICA_COUNT" '
        '--nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" --rdzv-id="$GANTRY_RDZV_ID" '
        '--rdzv-backend=static --rdzv-endpoint="$BEAKER_LEADER_REPLICA_HOSTNAME:$GANTRY_RDZV_PORT" '
        '--node-rank="$BEAKER_REPLICA_RANK" --rdzv-conf="read_timeout=420" '
        if nodes == 2
        else 'torchrun --nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
    )
    launch = launch_prefix + shlex.join([script, Path(args.output).name, *train_args])
    postflight = [
        shlex.join(["test", "-d", f"{args.output}/step{pre_decay}"]),
        shlex.join(["test", "-d", f"{args.output}/step{endpoint}"]),
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
        digest = hashlib.sha256(f"{args.output}:{args.target_epoch}".encode()).hexdigest()
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
    pre_decay, endpoint = FRONTIERS[args.target_epoch]
    spec = source_spec(args.method, args.simulated_sequences)
    script, source = extract_training(spec)
    train_args = training_arguments(source, args, pre_decay, endpoint)
    spec = build_spec(spec, script, train_args, args)
    shell = spec["tasks"][0]["arguments"][2]
    required = [args.source_checkpoint, f"step{pre_decay}", f"step{endpoint}", REVISION]
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"built spec missing guards: {missing}")
    if (
        args.start_epoch > 2
        and not args.conventional_transition
        and "recalibrate_second_moment_on_start=true" in shell
    ):
        raise RuntimeError("native local-update continuation must not recalibrate again")
    if args.conventional_transition and "recalibrate_second_moment_on_start=true" not in shell:
        raise RuntimeError("conventional-to-local transition must perform one-time recalibration")
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
