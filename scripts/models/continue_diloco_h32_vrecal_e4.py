#!/usr/bin/env python3
"""Build or submit the audited E4-only continuation for BS512-E1 DiLoCo H32."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


SOURCE_EXPERIMENT = "01KZWZQPBSB99S3H3SFNE141T8"
REVISION = "a1e6b71a5bbec5294ecf5498905f2ec850371ea7"
OUTPUT = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_"
    "poste1_diloco_h32_vrecal_dr_e2_lr1e-3_wd0.333_"
    "warmup48_vrecal-h32-dr-wd0p333-r02"
)
SOURCE_STEP = f"{OUTPUT}/step858"
SOURCE_REPLICA_MANIFEST = f"{OUTPUT}/step858-diloco-replicas/manifest.json"
PRE_DECAY_STEP = 1716
ENDPOINT_STEP = 1908
REPLICA_STEPS = (1287, PRE_DECAY_STEP)
REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
MIRROR = Path("reports/0802/data/wsd_batch_simulation_1b.js")


def fixed_interval_steps(source: int, boundaries: tuple[int, ...], interval: int = 32) -> list[int]:
    steps: list[int] = []
    cursor = source
    for boundary in boundaries:
        if boundary <= cursor:
            raise RuntimeError("DiLoCo boundaries must be strictly increasing")
        while cursor + interval < boundary:
            cursor += interval
            steps.append(cursor)
        cursor = boundary
        steps.append(cursor)
    return steps


OUTER_STEPS = fixed_interval_steps(858, (*REPLICA_STEPS, ENDPOINT_STEP))


def replace_unique(arguments: list[str], prefix: str, replacement: str) -> None:
    matches = [index for index, value in enumerate(arguments) if value.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {prefix}, found {len(matches)}")
    arguments[matches[0]] = replacement


def extract_training(source: dict) -> tuple[dict, str, str, list[str], int]:
    task = source["tasks"][0]
    for line in task["arguments"][2].splitlines():
        parts = shlex.split(line)
        if parts and parts[0] == "torchrun":
            nproc = int(parts[1].split("=", 1)[1])
            return task, parts[2], parts[3], parts[4:], nproc
    raise RuntimeError("source experiment has no torchrun command")


def assert_registered_before_submission() -> dict:
    report = json.loads(REPORT.read_text())
    matches = [
        run
        for run in report["runs"]
        if run.get("method") == "post_e1_diloco_h32_vrecal_dr"
        and run.get("startEpoch") == 2
        and run.get("chainThrough") == 4
        and run.get("lr") == "1e-3"
        and run.get("wd") == "0.333"
        and run.get("sourceCheckpoint") == SOURCE_STEP
        and run.get("status") in {"planned", "print-only-verified"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one registered E4 plan, found {len(matches)}")
    duplicates = [
        run
        for run in report["runs"]
        if run is not matches[0]
        and run.get("method") == "post_e1_diloco_h32_vrecal_dr"
        and run.get("chainThrough") == 4
        and run.get("lr") == "1e-3"
        and run.get("wd") == "0.333"
        and run.get("status") in {"submitted", "active", "running", "complete"}
    ]
    if duplicates:
        raise RuntimeError("DiLoCo E4 coordinate is already represented")
    return matches[0]


def build_spec() -> tuple[dict, dict]:
    source = json.loads(
        subprocess.run(
            ["beaker", "experiment", "spec", SOURCE_EXPERIMENT, "--format", "json"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    task, script, _old_name, arguments, nproc = extract_training(source)
    if nproc != 8:
        raise RuntimeError(f"expected one 8-GPU node, found nproc={nproc}")

    name = "bs512_dr_diloco_h32_vrecal_init=bs512e1_lr1e-3_wd0.333_e4"
    replacements = {
        "--save-folder=": f"--save-folder={OUTPUT}",
        "--trainer.max_duration=": "--trainer.max_duration={value: 4000000000, unit: tokens}",
        "--trainer.callbacks.wandb.name=": f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=": (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,repeated-data,wsd,"
            "bs512,simbs64,post_e1_diloco_h32_vrecal_dr,dynamic-repacking,post-e1,e4,wd0.333]"
        ),
        "--trainer.callbacks.checkpointer.fixed_steps=": (
            f"--trainer.callbacks.checkpointer.fixed_steps=[{PRE_DECAY_STEP}]"
        ),
        "--trainer.load_path=": f"--trainer.load_path={SOURCE_STEP}",
        "--trainer.load_trainer_state=": "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=": "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=": (
            "--trainer.reset_data_loader_state_on_load_path=false"
        ),
        "--train_module.batch_simulation.diloco_recalibrate_second_moment_on_start=": (
            "--train_module.batch_simulation.diloco_recalibrate_second_moment_on_start=false"
        ),
        "--train_module.batch_simulation.diloco_outer_steps=": (
            "--train_module.batch_simulation.diloco_outer_steps="
            + json.dumps(OUTER_STEPS, separators=(",", ":"))
        ),
        "--train_module.batch_simulation.diloco_replica_checkpoint_steps=": (
            "--train_module.batch_simulation.diloco_replica_checkpoint_steps="
            + json.dumps(REPLICA_STEPS, separators=(",", ":"))
        ),
    }
    for prefix, replacement in replacements.items():
        replace_unique(arguments, prefix, replacement)
    if not any(value.startswith("--trainer.prefer_explicit_load_path=") for value in arguments):
        arguments.append("--trainer.prefer_explicit_load_path=true")
    else:
        replace_unique(
            arguments,
            "--trainer.prefer_explicit_load_path=",
            "--trainer.prefer_explicit_load_path=true",
        )

    required = {
        "--lr=1e-3",
        "--train_module.optim.weight_decay=0.333",
        "--dynamic-repacking",
        "--train_module.batch_simulation.method=diloco",
        "--train_module.batch_simulation.diloco_inner_steps=32",
        "--train_module.batch_simulation.diloco_outer_lr=0.7",
        "--train_module.batch_simulation.diloco_outer_momentum=0.9",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.prefer_explicit_load_path=true",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    }
    missing = sorted(required - set(arguments))
    if missing:
        raise RuntimeError(f"missing audited settings: {missing}")

    commands = [
        "set -euo pipefail",
        f'test "$(git rev-parse HEAD)" = "{REVISION}"',
        shlex.join(["test", "-d", SOURCE_STEP]),
        shlex.join(["test", "-f", SOURCE_REPLICA_MANIFEST]),
        shlex.join(
            [
                "echo",
                "DILOCO_E4_PREFLIGHT source=step858 native_diloco_restore=true "
                "second_moment_recalibration=false H=32 outer_lr=0.7 outer_momentum=0.9 "
                "lr=1e-3 wd=0.333 dynamic_repacking=true",
            ]
        ),
        shlex.join(["python", script, name, "--dry-run", *arguments]),
        shlex.join(["torchrun", f"--nproc-per-node={nproc}", script, name, *arguments]),
        shlex.join(["test", "-d", f"{OUTPUT}/step{PRE_DECAY_STEP}"]),
        shlex.join(["test", "-d", f"{OUTPUT}/step{ENDPOINT_STEP}"]),
        *[
            shlex.join(["test", "-f", f"{OUTPUT}/step{step}-diloco-replicas/manifest.json"])
            for step in REPLICA_STEPS
        ],
        "echo DILOCO_E4_COMPLETE",
    ]
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["context"] = {"priority": "urgent", "minRuntime": 0, "autoResume": False}
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    for env in task.get("envVars", []):
        if env.get("name") == "GIT_REF":
            env["value"] = REVISION
        elif env.get("name") == "GIT_BRANCH":
            env["value"] = "sewonm/icsl"
    source["description"] = (
        "BS512-E1 initialized DiLoCo H32 v-recal DR WD0.333 E4-only native continuation "
        "from exact pre-decay step858."
    )
    stage = {
        "epoch": 4,
        "name": name,
        "output": OUTPUT,
        "sourceCheckpoint": SOURCE_STEP,
        "preDecayCheckpoint": f"{OUTPUT}/step{PRE_DECAY_STEP}",
        "endpointCheckpoint": f"{OUTPUT}/step{ENDPOINT_STEP}",
        "preDecayStep": PRE_DECAY_STEP,
        "endpointStep": ENDPOINT_STEP,
        "dilocoOuterSteps": OUTER_STEPS,
        "dilocoReplicaCheckpointSteps": list(REPLICA_STEPS),
        "dilocoReplicaCheckpointRoots": [
            f"{OUTPUT}/step{step}-diloco-replicas" for step in REPLICA_STEPS
        ],
        "secondMomentRecalibrationOnResume": False,
    }
    return source, stage


def experiment_id(output: str) -> str:
    matches = re.findall(r"\b01[A-Z0-9]{24}\b", output)
    if not matches:
        raise RuntimeError(f"could not parse experiment ID from {output!r}")
    return matches[0]


def register_submission(plan: dict, experiment: str, stage: dict) -> None:
    report = json.loads(REPORT.read_text())
    matches = [run for run in report["runs"] if run is not None and run == plan]
    if len(matches) != 1:
        raise RuntimeError("registered plan changed during submission")
    matches[0].update(
        {
            "status": "submitted",
            "healthStatus": "pending_startup",
            "beaker": experiment,
            "revision": REVISION,
            "stages": [stage],
            "reason": "Submitted audited E4-only native DiLoCo continuation from exact pre-decay step858; E8 remains conditional on E4 evidence.",
        }
    )
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    MIRROR.write_text(
        "window.ICSL_BATCH_SIMULATION_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = assert_registered_before_submission()
    spec, stage = build_spec()
    if not args.submit:
        print(json.dumps(spec, indent=2))
        return
    completed = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            "bs512-dr-diloco-h32-vrecal-init-bs512e1-wd0.333-e4",
            "--workspace",
            "ai2/flex2",
            "--priority",
            "urgent",
        ],
        check=True,
        input=json.dumps(spec),
        stdout=subprocess.PIPE,
        text=True,
    )
    experiment = experiment_id(completed.stdout)
    register_submission(plan, experiment, stage)
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
