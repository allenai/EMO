#!/usr/bin/env python3
"""Submit one guarded BS512 post-E1 Local-SGD or DiLoCo E2->E12 chain.

The launcher is intentionally narrow.  It accepts one of the three registered
LR1e-3 BS512 E1 pre-decay checkpoints, switches on dynamic repacking and one
local-update method, then evaluates E2, E4, E8, and E12 in one eight-GPU Beaker
job.  Every stage loads the preceding stage's exact pre-decay checkpoint so the
terminal WSD decay for an earlier endpoint is never used as training state for
the next endpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shlex
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

TOKENS_PER_EPOCH = 1_000_000_000
SEQUENCE_LENGTH = 4096
GLOBAL_SEQUENCES = 512
SIMULATED_SEQUENCES = 64
GLOBAL_BATCH_TOKENS = GLOBAL_SEQUENCES * SEQUENCE_LENGTH
SIMULATED_BATCH_TOKENS = SIMULATED_SEQUENCES * SEQUENCE_LENGTH
DECAY_FRACTION = 0.1
TARGETS = (2, 4, 8, 12)
LR = "1e-3"
REVISION_DEFAULT = "d0151b260e8c30f4251f43ec064ace57ebe9a7e8"
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models"
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
SIMULATION_REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
SIMULATION_MIRROR = Path("reports/0802/data/wsd_batch_simulation_1b.js")
BASELINE_REPORT = Path("reports/0802/data/wsd_batch_size_1b.json")
METHODS = (
    "post_e1_local_sgd_h4_dr",
    "post_e1_diloco_h500_dr",
    "post_e1_diloco_epoch_dr",
    "post_e1_diloco_h32_vrecal_dr",
    "post_e1_diloco_h32_bs64init_dr",
)
WEIGHT_DECAYS = ("0.1", "0.333", "1.0")


def total_steps(epoch: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / GLOBAL_BATCH_TOKENS)


def pre_decay_step(epoch: int) -> int:
    endpoint = total_steps(epoch)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--weight-decay", choices=WEIGHT_DECAYS, required=True)
    parser.add_argument("--revision", default=REVISION_DEFAULT)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--chain-through", type=int, choices=TARGETS, default=12)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    source_batch_sequences = (
        64 if args.method == "post_e1_diloco_h32_bs64init_dr" else GLOBAL_SEQUENCES
    )
    expected_source_step = pre_decay_step_for_batch(1, source_batch_sequences)
    if not args.source_checkpoint.endswith(f"/step{expected_source_step}"):
        parser.error(f"post-E1 transition requires exact /step{expected_source_step}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    return args


def numeric(value: object) -> Decimal:
    return Decimal(str(value))


def total_steps_for_batch(epoch: int, batch_sequences: int) -> int:
    batch_tokens = batch_sequences * SEQUENCE_LENGTH
    return math.ceil(epoch * TOKENS_PER_EPOCH / batch_tokens)


def pre_decay_step_for_batch(epoch: int, batch_sequences: int) -> int:
    endpoint = total_steps_for_batch(epoch, batch_sequences)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def get_spec(experiment: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def extract_training_command(task: dict[str, Any]) -> tuple[str, str, list[str], int]:
    arguments = task.get("arguments", [])
    if arguments[:2] != ["bash", "-lc"] or len(arguments) != 3:
        raise RuntimeError("expected a single bash task containing torchrun")
    for line in arguments[2].splitlines():
        parts = shlex.split(line)
        if not parts or parts[0] != "torchrun":
            continue
        nproc_args = [part for part in parts[1:] if part.startswith("--nproc-per-node=")]
        if len(nproc_args) != 1:
            raise RuntimeError("source torchrun has no unique nproc-per-node")
        nproc = int(nproc_args[0].split("=", 1)[1])
        script_index = parts.index(nproc_args[0]) + 1
        return parts[script_index], parts[script_index + 1], parts[script_index + 2 :], nproc
    raise RuntimeError("could not find source torchrun command")


def values_for(arguments: list[str], prefix: str) -> list[str]:
    return [value.split("=", 1)[1] for value in arguments if value.startswith(prefix)]


def unique_value(arguments: list[str], prefix: str) -> str:
    values = values_for(arguments, prefix)
    if len(values) != 1:
        raise RuntimeError(f"expected one {prefix!r} value, found {values}")
    return values[0]


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


def validate_source(args: argparse.Namespace, spec: dict[str, Any]) -> tuple[str, list[str], int]:
    if len(spec.get("tasks", [])) != 1:
        raise RuntimeError("source experiment must contain exactly one task")
    script, _, arguments, nproc = extract_training_command(spec["tasks"][0])
    if not script.endswith("olmo2-1B.py"):
        raise RuntimeError(f"unexpected source training script {script!r}")
    if nproc != 8:
        raise RuntimeError(f"source is not one eight-GPU node (nproc={nproc})")
    if numeric(unique_value(arguments, "--lr=")) != numeric(LR):
        raise RuntimeError("source LR does not match fixed LR1e-3")
    if numeric(unique_value(arguments, "--train_module.optim.weight_decay=")) != numeric(
        args.weight_decay
    ):
        raise RuntimeError("source WD does not match the requested coordinate")
    source_batch_sequences = (
        64 if args.method == "post_e1_diloco_h32_bs64init_dr" else GLOBAL_SEQUENCES
    )
    source_batch_tokens = source_batch_sequences * SEQUENCE_LENGTH
    if int(unique_value(arguments, "--data_loader.global_batch_size=")) != source_batch_tokens:
        raise RuntimeError("source global batch does not match the requested initialization")
    source_output = unique_value(arguments, "--save-folder=")
    source_step = pre_decay_step_for_batch(1, source_batch_sequences)
    if args.source_checkpoint != f"{source_output}/step{source_step}":
        raise RuntimeError("source checkpoint is not the source experiment's exact E1 checkpoint")

    baseline = json.loads(BASELINE_REPORT.read_text())
    parents = [
        run
        for run in baseline.get("batchSweeps", [])
        if run.get("beaker") == args.base_experiment
        and run.get("batchSequences") == source_batch_sequences
        and numeric(run.get("lr")) == numeric(LR)
        and numeric(run.get("wd")) == numeric(args.weight_decay)
        and (
            run.get("results", {}).get("1", {}).get("resumeCheckpoint")
            == args.source_checkpoint
            or f"{run.get('results', {}).get('1', {}).get('output')}/step{source_step}"
            == args.source_checkpoint
        )
    ]
    if len(parents) != 1 or parents[0].get("status") != "complete":
        raise RuntimeError("source is not the unique completed registered E1 parent")
    return script, arguments, nproc


def registered_plan(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(SIMULATION_REPORT.read_text())
    targets = [target for target in TARGETS if target <= args.chain_through]
    matches = [
        run
        for run in report.get("runs", [])
        if run.get("method") == args.method
        and run.get("startEpoch") == 2
        and run.get("chainThrough") == args.chain_through
        and run.get("targetLadder") == targets
        and run.get("batchSequences") == GLOBAL_SEQUENCES
        and run.get("simulatedBatchSequences") == SIMULATED_SEQUENCES
        and numeric(run.get("lr")) == numeric(LR)
        and numeric(run.get("wd")) == numeric(args.weight_decay)
        and run.get("parentExperiment") == args.base_experiment
        and run.get("sourceCheckpoint") == args.source_checkpoint
    ]
    if len(matches) != 1 or matches[0].get("status") not in {
        "planned",
        "print-only-verified",
    }:
        raise RuntimeError(
            "expected exactly one planned registered post-E1 chain; " f"found {len(matches)}"
        )
    attempted = [
        run
        for run in report.get("runs", [])
        if run is not matches[0]
        and run.get("method") == args.method
        and run.get("batchSequences") == GLOBAL_SEQUENCES
        and numeric(run.get("lr", "-1")) == numeric(LR)
        and numeric(run.get("wd", "-1")) == numeric(args.weight_decay)
        and (
            run.get("startEpoch") == 2
            or run.get("chainThrough") == args.chain_through
            or any(str(epoch) in run.get("results", {}) for epoch in targets)
        )
    ]
    if attempted:
        raise RuntimeError("this post-E1 local-update coordinate is already represented")
    return matches[0]


def method_arguments(
    method: str,
    *,
    diloco_outer_steps: tuple[int, ...] = (),
    diloco_replica_checkpoint_steps: tuple[int, ...] = (),
) -> list[str]:
    common = [
        f"--train_module.batch_simulation.global_batch_size={GLOBAL_BATCH_TOKENS}",
        f"--train_module.batch_simulation.simulated_batch_size={SIMULATED_BATCH_TOKENS}",
        "--train_module.batch_simulation.seed=12536",
    ]
    if method == "post_e1_local_sgd_h4_dr":
        return [
            "--train_module.batch_simulation.method=local_sgd",
            "--train_module.batch_simulation.local_sgd_sync_interval=4",
            *common,
        ]
    if method == "post_e1_diloco_h500_dr":
        return [
            "--train_module.batch_simulation.method=diloco",
            "--train_module.batch_simulation.diloco_inner_steps=500",
            "--train_module.batch_simulation.diloco_outer_lr=0.7",
            "--train_module.batch_simulation.diloco_outer_momentum=0.9",
            *common,
        ]
    if method == "post_e1_diloco_epoch_dr":
        if not diloco_outer_steps:
            raise RuntimeError("epoch-synchronized DiLoCo requires exact outer steps")
        if not diloco_replica_checkpoint_steps:
            raise RuntimeError("epoch-synchronized DiLoCo requires replica checkpoint steps")
        if not set(diloco_replica_checkpoint_steps).issubset(diloco_outer_steps):
            raise RuntimeError("replica checkpoint steps must be a subset of outer steps")
        return [
            "--train_module.batch_simulation.method=diloco",
            "--train_module.batch_simulation.diloco_inner_steps=500",
            "--train_module.batch_simulation.diloco_outer_lr=0.7",
            "--train_module.batch_simulation.diloco_outer_momentum=0.9",
            "--train_module.batch_simulation.diloco_outer_steps="
            + json.dumps(diloco_outer_steps, separators=(",", ":")),
            "--train_module.batch_simulation.diloco_replica_checkpoint_steps="
            + json.dumps(diloco_replica_checkpoint_steps, separators=(",", ":")),
            *common,
        ]
    if method == "post_e1_diloco_h32_vrecal_dr":
        if not diloco_outer_steps:
            raise RuntimeError("H=32 DiLoCo requires exact outer steps")
        if not diloco_replica_checkpoint_steps:
            raise RuntimeError("H=32 DiLoCo requires replica checkpoint steps")
        if not set(diloco_replica_checkpoint_steps).issubset(diloco_outer_steps):
            raise RuntimeError("replica checkpoint steps must be a subset of outer steps")
        return [
            "--train_module.batch_simulation.method=diloco",
            "--train_module.batch_simulation.diloco_inner_steps=32",
            "--train_module.batch_simulation.diloco_outer_lr=0.7",
            "--train_module.batch_simulation.diloco_outer_momentum=0.9",
            "--train_module.batch_simulation.diloco_recalibrate_second_moment_on_start=true",
            "--train_module.batch_simulation.diloco_outer_steps="
            + json.dumps(diloco_outer_steps, separators=(",", ":")),
            "--train_module.batch_simulation.diloco_replica_checkpoint_steps="
            + json.dumps(diloco_replica_checkpoint_steps, separators=(",", ":")),
            *common,
        ]
    if method == "post_e1_diloco_h32_bs64init_dr":
        if not diloco_outer_steps:
            raise RuntimeError("BS64-initialized H=32 DiLoCo requires exact outer steps")
        if not diloco_replica_checkpoint_steps:
            raise RuntimeError("BS64-initialized H=32 DiLoCo requires replica checkpoint steps")
        if not set(diloco_replica_checkpoint_steps).issubset(diloco_outer_steps):
            raise RuntimeError("replica checkpoint steps must be a subset of outer steps")
        return [
            "--train_module.batch_simulation.method=diloco",
            "--train_module.batch_simulation.diloco_inner_steps=32",
            "--train_module.batch_simulation.diloco_outer_lr=0.7",
            "--train_module.batch_simulation.diloco_outer_momentum=0.9",
            "--train_module.batch_simulation.diloco_outer_steps="
            + json.dumps(diloco_outer_steps, separators=(",", ":")),
            "--train_module.batch_simulation.diloco_replica_checkpoint_steps="
            + json.dumps(diloco_replica_checkpoint_steps, separators=(",", ":")),
            *common,
        ]
    raise RuntimeError(f"unsupported method {method}")


def fixed_interval_outer_steps(
    source_step: int,
    boundaries: tuple[int, ...],
    *,
    interval: int,
) -> tuple[int, ...]:
    """Return full H-step rounds plus an exact aggregation at each semantic boundary."""
    steps: list[int] = []
    cursor = source_step
    for boundary in boundaries:
        if boundary <= cursor:
            raise RuntimeError("DiLoCo outer-step boundaries must be strictly increasing")
        while cursor + interval < boundary:
            cursor += interval
            steps.append(cursor)
        steps.append(boundary)
        cursor = boundary
    return tuple(steps)


def bs64_initialized_e2_steps() -> tuple[int, int, int]:
    """Map the BS64 E1 token position onto a BS512 continuation without remapping Adam time."""
    source_step = pre_decay_step_for_batch(1, 64)
    source_tokens = source_step * 64 * SEQUENCE_LENGTH
    decay_start_tokens = round(2 * TOKENS_PER_EPOCH * (1.0 - DECAY_FRACTION))
    stable_continuation_steps = math.ceil(
        (decay_start_tokens - source_tokens) / GLOBAL_BATCH_TOKENS
    )
    total_continuation_steps = math.ceil(
        (2 * TOKENS_PER_EPOCH - source_tokens) / GLOBAL_BATCH_TOKENS
    )
    pre_decay = source_step + stable_continuation_steps - 1
    endpoint = source_step + total_continuation_steps
    return source_step, pre_decay, endpoint


def stage(
    base_arguments: list[str],
    args: argparse.Namespace,
    script: str,
    *,
    epoch: int,
    previous_epoch: int,
    load_path: str,
) -> tuple[str, str, list[str]]:
    method_tags = {
        "post_e1_local_sgd_h4_dr": "poste1_ls_h4_dr",
        "post_e1_diloco_h500_dr": "poste1_diloco_h500_dr",
        "post_e1_diloco_epoch_dr": "poste1_diloco_epoch_dr",
        "post_e1_diloco_h32_vrecal_dr": "poste1_diloco_h32_vrecal_dr",
        "post_e1_diloco_h32_bs64init_dr": "poste1_diloco_h32_bs64init_dr",
    }
    method_tag = method_tags[args.method]
    name = (
        "dense_1b_step1_0802_repeated_dclm1b_wsd_bs512_"
        f"{method_tag}_e{epoch}_lr1e-3_wd{args.weight_decay}_warmup48_{args.suffix}"
    )
    output = f"{OUTPUT_ROOT}/{name}"
    excluded_prefixes = (
        "--load_path=",
        "--load_trainer_state=",
        "--trainer.load_path=",
        "--trainer.load_trainer_state=",
        "--trainer.load_optim_state=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--train_module.batch_simulation.",
        "--data_loader.restore_data_order_from_state=",
        "--data_loader.ignore_fingerprint_mismatch=",
        "--train_module.validate_optimizer_hyperparameters_on_load=",
        "--trainer.callbacks.checkpointer.save_interval=",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
    )
    arguments = [
        value
        for value in base_arguments
        if value
        not in {
            "--dynamic-repacking",
            "--fixed-data-order",
            "--no-data-shuffle",
            "--batch-shuffling",
        }
        and not value.startswith(excluded_prefixes)
    ]
    if args.method == "post_e1_diloco_h32_bs64init_dr":
        if previous_epoch != 1 or epoch != 2:
            raise RuntimeError("BS64-initialized H=32 DiLoCo is currently guarded to E2 only")
        _, target_pre_decay, target_endpoint = bs64_initialized_e2_steps()
        scheduler = (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: tokens, "
            "warmup: 100663296, decay_fraction: 0.1}"
        )
    else:
        target_pre_decay = pre_decay_step(epoch)
        target_endpoint = total_steps(epoch)
        scheduler = (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, "
            "warmup: 48, decay_fraction: 0.1}"
        )
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--dataset.mix=", "--dataset.mix=null"),
        ("--dataset.subset_manifest=", f"--dataset.subset_manifest={TRAIN_MANIFEST}"),
        (
            "--trainer.max_duration=",
            f"--trainer.max_duration={{value: {epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags="
            + f"[pretraining,step1,0802,repeated-data,wsd,bs512,simbs64,{args.method},dynamic-repacking,post-e1,e{epoch},wd{args.weight_decay}]",
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{target_pre_decay}]",
        ),
        (
            "--data_loader.global_batch_size=",
            f"--data_loader.global_batch_size={GLOBAL_BATCH_TOKENS}",
        ),
        ("--data_loader.seed=", "--data_loader.seed=0"),
        (
            "--data_loader.restore_data_order_from_state=",
            "--data_loader.restore_data_order_from_state=false",
        ),
        (
            "--data_loader.ignore_fingerprint_mismatch=",
            "--data_loader.ignore_fingerprint_mismatch=true",
        ),
        (
            "--train_module.rank_microbatch_size=",
            f"--train_module.rank_microbatch_size={4 * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.scheduler=",
            scheduler,
        ),
        (
            "--train_module.optim.weight_decay=",
            f"--train_module.optim.weight_decay={args.weight_decay}",
        ),
        ("--lr=", f"--lr={LR}"),
        ("--trainer.load_path=", f"--trainer.load_path={load_path}"),
        ("--trainer.load_trainer_state=", "--trainer.load_trainer_state=true"),
        ("--trainer.load_optim_state=", "--trainer.load_optim_state=true"),
        (
            "--trainer.prefer_explicit_load_path=",
            "--trainer.prefer_explicit_load_path="
            + ("true" if previous_epoch > 1 else "false"),
        ),
        (
            "--trainer.reset_data_loader_state_on_load_path=",
            "--trainer.reset_data_loader_state_on_load_path="
            + (
                "true"
                if args.method == "post_e1_diloco_h32_bs64init_dr"
                else "false"
            ),
        ),
        (
            "--train_module.validate_optimizer_hyperparameters_on_load=",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        ),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)
    integer_epoch_frontiers = (target_pre_decay,)
    # The terminal endpoint also needs one explicit aggregation so evaluation observes a
    # coherent model. The next stage still resumes from the synchronized pre-decay frontier,
    # so this terminal decay aggregation cannot affect subsequent training.
    if args.method == "post_e1_diloco_h32_bs64init_dr":
        source_step, _, _ = bs64_initialized_e2_steps()
        diloco_outer_steps = fixed_interval_outer_steps(
            source_step,
            integer_epoch_frontiers + (target_endpoint,),
            interval=32,
        )
    elif args.method == "post_e1_diloco_h32_vrecal_dr":
        diloco_outer_steps = fixed_interval_outer_steps(
            pre_decay_step(previous_epoch),
            integer_epoch_frontiers + (total_steps(epoch),),
            interval=32,
        )
    else:
        diloco_outer_steps = integer_epoch_frontiers + (total_steps(epoch),)
    arguments.extend(
        [
            "--dynamic-repacking",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            *method_arguments(
                args.method,
                diloco_outer_steps=diloco_outer_steps,
                diloco_replica_checkpoint_steps=integer_epoch_frontiers,
            ),
        ]
    )
    return name, output, arguments


def build_spec(
    base_spec: dict[str, Any],
    args: argparse.Namespace,
    script: str,
    base_arguments: list[str],
    nproc: int,
    targets: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    commands = ["set -euo pipefail"]
    stage_records: list[dict[str, Any]] = []
    load_path = args.source_checkpoint
    previous_epoch = 1
    for epoch in targets:
        name, output, arguments = stage(
            base_arguments,
            args,
            script,
            epoch=epoch,
            previous_epoch=previous_epoch,
            load_path=load_path,
        )
        if args.method == "post_e1_diloco_h32_bs64init_dr":
            source_step, pre_decay, endpoint = bs64_initialized_e2_steps()
            integer_epoch_frontiers = (pre_decay,)
            recorded_outer_steps = fixed_interval_outer_steps(
                source_step,
                integer_epoch_frontiers + (endpoint,),
                interval=32,
            )
        else:
            pre_decay = pre_decay_step(epoch)
            endpoint = total_steps(epoch)
            integer_epoch_frontiers = tuple(
                pre_decay_step(frontier_epoch)
                for frontier_epoch in range(previous_epoch + 1, epoch + 1)
            )
        if args.method == "post_e1_diloco_h32_bs64init_dr":
            pass
        elif args.method == "post_e1_diloco_h32_vrecal_dr":
            recorded_outer_steps = fixed_interval_outer_steps(
                pre_decay_step(previous_epoch),
                integer_epoch_frontiers + (endpoint,),
                interval=32,
            )
        else:
            recorded_outer_steps = integer_epoch_frontiers + (endpoint,)
        commands.extend(
            [
                shlex.join(["test", "-d", load_path]),
                shlex.join(["test", "-f", TRAIN_MANIFEST]),
                shlex.join(["test", "!", "-e", output]),
                shlex.join(
                    [
                        "echo",
                        (
                            f"POST_E1_LOCAL_UPDATE_PREFLIGHT method={args.method} epoch={epoch} "
                            f"source={load_path} pre_decay_step={pre_decay} "
                            f"endpoint_step={endpoint} dynamic_repacking=true "
                            "load_trainer_state=true load_optim_state=true"
                        ),
                    ]
                ),
                shlex.join(["python", script, name, "--dry-run", *arguments]),
                shlex.join(["torchrun", f"--nproc-per-node={nproc}", script, name, *arguments]),
                shlex.join(["test", "-d", f"{output}/step{pre_decay}"]),
                shlex.join(["test", "-d", f"{output}/step{endpoint}"]),
                *(
                    [
                        shlex.join(
                            [
                                "test",
                                "-f",
                                f"{output}/step{frontier}-diloco-replicas/manifest.json",
                            ]
                        )
                        for frontier in integer_epoch_frontiers
                    ]
                    if args.method
                    in {
                        "post_e1_diloco_epoch_dr",
                        "post_e1_diloco_h32_vrecal_dr",
                        "post_e1_diloco_h32_bs64init_dr",
                    }
                    else []
                ),
                shlex.join(
                    [
                        "echo",
                        f"POST_E1_LOCAL_UPDATE_STAGE_COMPLETE method={args.method} epoch={epoch}",
                    ]
                ),
            ]
        )
        stage_records.append(
            {
                "epoch": epoch,
                "name": name,
                "output": output,
                "sourceCheckpoint": load_path,
                "preDecayCheckpoint": f"{output}/step{pre_decay}",
                "endpointCheckpoint": f"{output}/step{endpoint}",
                "preDecayStep": pre_decay,
                "endpointStep": endpoint,
                **(
                    {
                        "dilocoOuterSteps": list(recorded_outer_steps),
                        "dilocoReplicaCheckpointSteps": list(integer_epoch_frontiers),
                        "dilocoReplicaCheckpointRoots": [
                            f"{output}/step{frontier}-diloco-replicas"
                            for frontier in integer_epoch_frontiers
                        ],
                    }
                    if args.method
                    in {
                        "post_e1_diloco_epoch_dr",
                        "post_e1_diloco_h32_vrecal_dr",
                        "post_e1_diloco_h32_bs64init_dr",
                    }
                    else {}
                ),
            }
        )
        load_path = f"{output}/step{pre_decay}"
        previous_epoch = epoch
    commands.append(
        shlex.join(
            [
                "echo",
                (
                    "POST_E1_LOCAL_UPDATE_CHAIN_COMPLETE "
                    f"method={args.method} targets={','.join(str(target) for target in targets)}"
                ),
            ]
        )
    )
    task["name"] = "main"
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": 0, "autoResume": False}
    task["hostNetworking"] = False
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    task.pop("synchronizedStartTimeout", None)
    task["envVars"] = [
        item for item in task.get("envVars", []) if item.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    env_updates = {"GIT_REF": args.revision, "GIT_BRANCH": "sewonm/icsl", "NUM_NODES": "1"}
    for key, value in env_updates.items():
        for item in task["envVars"]:
            if item.get("name") == key:
                item["value"] = value
                break
        else:
            task["envVars"].append({"name": key, "value": value})
    spec.pop("description", None)
    return spec, stage_records


def update_registry(plan: dict[str, Any], experiment: str, stages: list[dict[str, Any]]) -> None:
    report = json.loads(SIMULATION_REPORT.read_text())
    matches = [
        run
        for run in report["runs"]
        if run.get("method") == plan["method"]
        and numeric(run.get("wd")) == numeric(plan["wd"])
        and run.get("sourceCheckpoint") == plan["sourceCheckpoint"]
        and run.get("status") in {"planned", "print-only-verified"}
    ]
    if len(matches) != 1:
        raise RuntimeError("registered plan changed during submission")
    matches[0].update(
        {
            "status": "submitted",
            "beaker": experiment,
            "revision": plan.get("revision", REVISION_DEFAULT),
            "stages": stages,
            "reason": "Submitted one guarded eight-GPU post-E1 dynamic-repacking chain through "
            + "/".join(f"E{stage['epoch']}" for stage in stages)
            + ".",
        }
    )
    report["updated"] = "2026-08-11"
    SIMULATION_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    SIMULATION_MIRROR.write_text(
        "window.ICSL_BATCH_SIMULATION_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def experiment_id(output: str) -> str:
    matches = re.findall(r"\b01[A-Z0-9]{24}\b", output)
    if not matches:
        raise RuntimeError(f"could not parse Beaker experiment ID from: {output!r}")
    return matches[0]


def main() -> None:
    args = parse_args()
    plan = registered_plan(args)
    source_spec = get_spec(args.base_experiment)
    script, base_arguments, nproc = validate_source(args, source_spec)
    targets = tuple(target for target in TARGETS if target <= args.chain_through)
    spec, stages = build_spec(source_spec, args, script, base_arguments, nproc, targets)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    completed = subprocess.run(
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
    beaker = experiment_id(completed.stdout)
    update_registry(plan, beaker, stages)
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
