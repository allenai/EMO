#!/usr/bin/env python3
"""Submit one guarded BS1024-DR E2 -> E4 local-update diagnostic.

This launcher is deliberately restricted to the four authorized coordinates:
LocalSGD or DiLoCo, H=32, simulated BS64 or BS256, LR1e-3, WD0.333.
Every coordinate restores the exact conventional BS1024-DR E2 pre-decay
checkpoint at step428 and writes E4 pre-decay step858 plus endpoint step954.
"""

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


SOURCE_EXPERIMENT = "01KZRTKNSSX9H1DPZ7NHJSM28F"
SOURCE_CHECKPOINT = (
    "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/"
    "bs1024_dr_lr1e-3_wd0.333/step428"
)
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
REPORT_JS = REPORT.with_suffix(".js")
REVISION = "b2920764d337b195842d592a59f8125637b067ac"
SEQUENCE_LENGTH = 4096
GLOBAL_SEQUENCES = 1024
GLOBAL_BATCH_TOKENS = GLOBAL_SEQUENCES * SEQUENCE_LENGTH
SOURCE_STEP = 428
PRE_DECAY_STEP = 858
ENDPOINT_STEP = 954
RANK_MICROBATCH_SEQUENCES = 4
LR = "1e-3"
WD = "0.333"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("local_sgd", "diloco"), required=True)
    parser.add_argument("--simulated-sequences", type=int, choices=(64, 256), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    return args


def method_key(method: str, simulated_sequences: int) -> str:
    if method == "local_sgd":
        base = "post_e2_local_sgd_h32_bs1024_dr"
    else:
        base = "post_e2_diloco_h32_vrecal_bs1024_dr"
    return f"{base}_simbs{simulated_sequences}"


def output_path(method: str, simulated_sequences: int) -> str:
    if method == "local_sgd":
        tag = "local_sgd_h32"
    else:
        tag = "diloco_h32_vrecal"
    return (
        f"{OUTPUT_ROOT}/bs1024_dr_{tag}_simbs{simulated_sequences}_"
        f"init=bs1024e2_lr1e-3_wd0.333"
    )


def load_report_plan(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(REPORT.read_text())
    matches = [
        run
        for run in report.get("runs", [])
        if run.get("method") == method_key(args.method, args.simulated_sequences)
        and run.get("batchSequences") == GLOBAL_SEQUENCES
        and run.get("simulatedBatchSequences") == args.simulated_sequences
        and run.get("lr") == LR
        and run.get("wd") == WD
        and run.get("sourceCheckpoint") == SOURCE_CHECKPOINT
        and run.get("targetEpoch") == 4
        and run.get("status") in {"planned", "print-only-verified"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one registered plan, found {len(matches)}")
    for run in report.get("runs", []):
        if run is matches[0]:
            continue
        if (
            run.get("method") == method_key(args.method, args.simulated_sequences)
            and run.get("batchSequences") == GLOBAL_SEQUENCES
            and run.get("simulatedBatchSequences") == args.simulated_sequences
            and run.get("lr") == LR
            and run.get("wd") == WD
            and run.get("targetEpoch") == 4
            and run.get("status")
            in {"submitted", "scheduled", "queued", "running", "complete"}
        ):
            raise RuntimeError("refusing duplicate active/completed coordinate")
    return matches[0]


def get_source_spec() -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", SOURCE_EXPERIMENT, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def extract_training(spec: dict[str, Any]) -> tuple[str, list[str]]:
    tasks = spec.get("tasks", [])
    if not tasks:
        raise RuntimeError("source experiment has no tasks")
    task = tasks[0]
    shell_args = task.get("arguments", [])
    if shell_args[:2] != ["bash", "-lc"] or len(shell_args) != 3:
        raise RuntimeError("unexpected source task shell")
    for line in shell_args[2].splitlines():
        if not line.startswith("torchrun "):
            continue
        parts = shlex.split(line)
        indices = [i for i, value in enumerate(parts) if value.endswith("olmo2-1B.py")]
        if parts and parts[0] == "torchrun" and len(indices) == 1:
            index = indices[0]
            return parts[index], parts[index + 2 :]
    raise RuntimeError("could not extract source torchrun training arguments")


def values_for(arguments: list[str], prefix: str) -> list[str]:
    return [value.split("=", 1)[1] for value in arguments if value.startswith(prefix)]


def unique_value(arguments: list[str], prefix: str) -> str:
    values = values_for(arguments, prefix)
    if len(values) != 1:
        raise RuntimeError(f"expected one {prefix}, found {values}")
    return values[0]


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    output: list[str] = []
    seen = False
    for value in arguments:
        if value.startswith(prefix):
            if not seen:
                output.append(replacement)
                seen = True
        else:
            output.append(value)
    if not seen:
        output.append(replacement)
    return output


def validate_source(arguments: list[str]) -> None:
    checks = {
        "--data_loader.global_batch_size=": str(GLOBAL_BATCH_TOKENS),
        "--lr=": LR,
        "--train_module.optim.weight_decay=": WD,
    }
    for prefix, expected in checks.items():
        actual = unique_value(arguments, prefix)
        if actual != expected:
            raise RuntimeError(f"source {prefix} is {actual}, expected {expected}")
    if "--dynamic-repacking" not in arguments:
        raise RuntimeError("source is not dynamic repacking")


def diloco_outer_steps() -> list[int]:
    steps: list[int] = []
    cursor = SOURCE_STEP
    for boundary in (PRE_DECAY_STEP, ENDPOINT_STEP):
        while cursor + 32 < boundary:
            cursor += 32
            steps.append(cursor)
        steps.append(boundary)
        cursor = boundary
    return steps


def training_arguments(
    source_arguments: list[str], method: str, simulated_sequences: int, output: str
) -> list[str]:
    excluded = (
        "--load_path=",
        "--load_trainer_state=",
        "--trainer.load_path=",
        "--trainer.load_trainer_state=",
        "--trainer.load_optim_state=",
        "--trainer.prefer_explicit_load_path=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--train_module.validate_optimizer_hyperparameters_on_load=",
        "--train_module.batch_simulation.",
        "--trainer.callbacks.checkpointer.save_interval=",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
    )
    arguments = [value for value in source_arguments if not value.startswith(excluded)]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--trainer.max_duration=", "--trainer.max_duration={value: 4000000000, unit: tokens}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={Path(output).name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            f"[pretraining,step1,local-update,bs1024,simbs{simulated_sequences},"
            f"{method},h32,dynamic-repacking,post-e2,e4,lr1e-3,wd0p333]",
        ),
        ("--trainer.callbacks.checkpointer.fixed_steps=", "--trainer.callbacks.checkpointer.fixed_steps=[858]"),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={GLOBAL_BATCH_TOKENS}"),
        ("--data_loader.restore_data_order_from_state=", "--data_loader.restore_data_order_from_state=true"),
        ("--data_loader.ignore_fingerprint_mismatch=", "--data_loader.ignore_fingerprint_mismatch=false"),
        (
            "--train_module.rank_microbatch_size=",
            f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.scheduler=",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 24, decay_fraction: 0.1}",
        ),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={WD}"),
        ("--lr=", f"--lr={LR}"),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    arguments.extend(
        [
            f"--trainer.load_path={SOURCE_CHECKPOINT}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.prefer_explicit_load_path=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            f"--train_module.batch_simulation.global_batch_size={GLOBAL_BATCH_TOKENS}",
            "--train_module.batch_simulation.simulated_batch_size="
            f"{simulated_sequences * SEQUENCE_LENGTH}",
            "--train_module.batch_simulation.seed=12536",
        ]
    )
    if method == "local_sgd":
        arguments.extend(
            [
                "--train_module.batch_simulation.method=local_sgd",
                "--train_module.batch_simulation.local_sgd_sync_interval=32",
                "--train_module.batch_simulation.recalibrate_second_moment_on_start=true",
            ]
        )
    else:
        arguments.extend(
            [
                "--train_module.batch_simulation.method=diloco",
                "--train_module.batch_simulation.diloco_inner_steps=32",
                "--train_module.batch_simulation.diloco_outer_lr=0.7",
                "--train_module.batch_simulation.diloco_outer_momentum=0.9",
                "--train_module.batch_simulation.recalibrate_second_moment_on_start=true",
                "--train_module.batch_simulation.diloco_outer_steps="
                + json.dumps(diloco_outer_steps(), separators=(",", ":")),
            ]
        )
    return arguments


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    for env in task["envVars"]:
        if env.get("name") == name:
            env["value"] = value
            return
    task["envVars"].append({"name": name, "value": value})


def build_spec(
    source_spec: dict[str, Any], args: argparse.Namespace, script: str, train_args: list[str]
) -> tuple[dict[str, Any], str, int]:
    spec = copy.deepcopy(source_spec)
    task = copy.deepcopy(spec["tasks"][0])
    spec["tasks"] = [task]
    output = output_path(args.method, args.simulated_sequences)
    nodes = 2 if args.simulated_sequences == 64 else 1
    replicas = GLOBAL_SEQUENCES // args.simulated_sequences
    shard_degree = nodes * 8 // replicas
    ratio = replicas
    checks = [
        "set -euo pipefail",
        shlex.join(["test", "-d", SOURCE_CHECKPOINT]),
        shlex.join(["test", "-f", TRAIN_MANIFEST]),
        shlex.join(["test", "!", "-e", output]),
        f'test "$(git rev-parse HEAD)" = "{REVISION}"',
        shlex.join(
            [
                "echo",
                (
                    f"BS1024_LOCAL_UPDATE_PREFLIGHT method={args.method} h=32 "
                    f"simbs={args.simulated_sequences} replicas={replicas} nodes={nodes} "
                    f"shard_degree={shard_degree} source_step=428 target_pre_decay=858 "
                    f"endpoint=954 lr={LR} wd={WD} rank_mb=4 dr=true "
                    "load_trainer=true load_optim=true explicit_load=true "
                    + (
                        f"v_recal=true v_recal_ratio={ratio} outer_lr=0.7 outer_momentum=0.9"
                        if args.method == "diloco"
                        else "v_recal=false"
                    )
                ),
            ]
        ),
        shlex.join(["python", script, Path(output).name, "--dry-run", *train_args]),
    ]
    preflight = "\n".join(checks)
    if nodes == 1:
        launch = (
            'torchrun --nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
            + shlex.join([script, Path(output).name, *train_args])
        )
    else:
        launch = (
            'torchrun --nnodes="$BEAKER_REPLICA_COUNT:$BEAKER_REPLICA_COUNT" '
            '--nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
            '--rdzv-id="$GANTRY_RDZV_ID" --rdzv-backend=static '
            '--rdzv-endpoint="$BEAKER_LEADER_REPLICA_HOSTNAME:$GANTRY_RDZV_PORT" '
            '--node-rank="$BEAKER_REPLICA_RANK" --rdzv-conf="read_timeout=420" '
            + shlex.join([script, Path(output).name, *train_args])
        )
    postflight = [
        shlex.join(["test", "-d", f"{output}/step858"]),
        shlex.join(["test", "-d", f"{output}/step954"]),
    ]
    task["arguments"] = [
        "bash",
        "-lc",
        (
            "set -euo pipefail\n"
            'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then\n'
            f"{preflight}\n"
            "fi\n"
            f"{launch}\n"
            'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then\n'
            + "\n".join(postflight)
            + "\nfi"
        ),
    ]
    blocked = {"GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "GANTRY_USE_TORCHRUN", "NUM_NODES"}
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in blocked
        and not (str(env.get("name", "")).startswith("BEAKER_") and env.get("name") != "BEAKER_TOKEN")
    ]
    set_env(task, "GIT_REF", REVISION)
    set_env(task, "GIT_BRANCH", "sewonm/icsl")
    set_env(task, "NUM_NODES", str(nodes))
    rendezvous = hashlib.sha256(output.encode()).hexdigest()
    if nodes > 1:
        set_env(task, "GANTRY_RDZV_ID", rendezvous[:12])
        set_env(task, "GANTRY_RDZV_PORT", str(29_000 + int(rendezvous[:8], 16) % 1_000))
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    if nodes > 1:
        task["replicas"] = nodes
        task["leaderSelection"] = True
        task["hostNetworking"] = True
        task["propagateFailure"] = True
        task["propagatePreemption"] = True
        task["synchronizedStartTimeout"] = "90m"
    else:
        for field in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
            task.pop(field, None)
        task["hostNetworking"] = False
        task["propagateFailure"] = False
        task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec, output, nodes


def validate_built_spec(
    spec: dict[str, Any], output: str, nodes: int, args: argparse.Namespace
) -> None:
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("built spec must contain exactly one replicated task")
    task = spec["tasks"][0]
    if task.get("resources", {}).get("gpuCount") != 8:
        raise RuntimeError("built spec must request eight GPUs per node")
    if nodes == 2:
        if task.get("replicas") != 2 or not task.get("leaderSelection"):
            raise RuntimeError("simBS64 must use two synchronized 8-GPU nodes")
    elif "replicas" in task:
        raise RuntimeError("simBS256 must use one 8-GPU node")
    shell = task.get("arguments", [None, None, ""])[2]
    required = [
        SOURCE_CHECKPOINT,
        output,
        f"--trainer.load_path={SOURCE_CHECKPOINT}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.prefer_explicit_load_path=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
        f"--data_loader.global_batch_size={GLOBAL_BATCH_TOKENS}",
        f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}",
        f"--train_module.batch_simulation.global_batch_size={GLOBAL_BATCH_TOKENS}",
        "--train_module.batch_simulation.simulated_batch_size="
        f"{args.simulated_sequences * SEQUENCE_LENGTH}",
        "--trainer.callbacks.checkpointer.fixed_steps=[858]",
        "--trainer.max_duration={value: 4000000000, unit: tokens}",
        "--dynamic-repacking",
        "warmup: 24",
        "decay_fraction: 0.1",
        "step858",
        "step954",
    ]
    if args.method == "local_sgd":
        required += [
            "--train_module.batch_simulation.method=local_sgd",
            "--train_module.batch_simulation.local_sgd_sync_interval=32",
            "--train_module.batch_simulation.recalibrate_second_moment_on_start=true",
        ]
        forbidden = ["diloco_outer_lr"]
    else:
        required += [
            "--train_module.batch_simulation.method=diloco",
            "--train_module.batch_simulation.diloco_inner_steps=32",
            "--train_module.batch_simulation.diloco_outer_lr=0.7",
            "--train_module.batch_simulation.diloco_outer_momentum=0.9",
            "--train_module.batch_simulation.recalibrate_second_moment_on_start=true",
            json.dumps(diloco_outer_steps(), separators=(",", ":")),
        ]
        forbidden = ["local_sgd_sync_interval"]
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"built spec is missing guards/config: {missing}")
    present_forbidden = [value for value in forbidden if value in shell]
    if present_forbidden:
        raise RuntimeError(f"built spec contains incompatible config: {present_forbidden}")
    if "/step477" in shell:
        raise RuntimeError("built spec must never load the decayed E2 endpoint step477")


def experiment_id(output: str) -> str:
    matches = re.findall(r"\b01[A-Z0-9]{24}\b", output)
    if not matches:
        raise RuntimeError(f"could not parse experiment ID from {output!r}")
    return matches[0]


def update_registry(
    plan: dict[str, Any], experiment: str, output: str, nodes: int, args: argparse.Namespace
) -> None:
    report = json.loads(REPORT.read_text())
    matches = [run for run in report["runs"] if run.get("planId") == plan.get("planId")]
    if len(matches) != 1 or matches[0].get("status") not in {"planned", "print-only-verified"}:
        raise RuntimeError("registered plan changed during submission")
    matches[0].update(
        {
            "status": "submitted",
            "beaker": experiment,
            "revision": REVISION,
            "output": output,
            "nodeCount": nodes,
            "gpuCount": nodes * 8,
            "preDecayCheckpoint": f"{output}/step858",
            "endpointCheckpoint": f"{output}/step954",
            "reason": "Submitted guarded BS1024-DR E2 step428 to E4 local-update continuation.",
        }
    )
    report["updated"] = "2026-08-13"
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_BATCH_SIMULATION_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    plan = load_report_plan(args)
    source_spec = get_source_spec()
    script, source_arguments = extract_training(source_spec)
    validate_source(source_arguments)
    output = output_path(args.method, args.simulated_sequences)
    train_args = training_arguments(source_arguments, args.method, args.simulated_sequences, output)
    spec, output, nodes = build_spec(source_spec, args, script, train_args)
    validate_built_spec(spec, output, nodes, args)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    created = subprocess.run(
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
        stdout=subprocess.PIPE,
        text=True,
    )
    experiment = experiment_id(created.stdout)
    update_registry(plan, experiment, output, nodes, args)
    print(created.stdout, end="")


if __name__ == "__main__":
    main()
