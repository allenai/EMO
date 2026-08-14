#!/usr/bin/env python3
"""Submit one guarded Dense-1B data-loader endpoint.

The launcher is deliberately narrow. It manages only the registered repeated-pool
dynamic-repacking and fixed-order study registered in
``reports/0802/data/wsd_data_loader_1b.json``. E2+ submissions must resume an exact
LR/WD-matched pre-decay checkpoint. A separately registered E1 bootstrap may start
fresh with ordinary packing when no exact coordinate exists. Every submission uses
rank microbatch eight and writes one WSD endpoint before the monitor decides which
coordinate may continue.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
MODEL_ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"
SEQUENCE_LENGTH = 4096
TOKENS_PER_EPOCH = 1_000_000_000
DECAY_FRACTION = 0.1
DEFAULT_LEARNING_RATE = "1e-3"
# After E4, guarded trajectories advance in four-epoch increments. Keep the
# launcher ahead of the currently displayed frontier so a strictly improving
# endpoint can be continued without weakening any exact-resume checks.
TARGETS = (1, 2, 4, *range(8, 65, 4))
PREDECESSOR = {2: 1, 4: 2, **{epoch: epoch - 4 for epoch in range(8, 65, 4)}}
GLOBAL_BATCHES = (64, 128, 256, 512, 1024)
DEFAULT_NODES = 4
GPUS_PER_NODE = 8
RANK_MICROBATCH_SEQUENCES = 8
TRAIN_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
ATTEMPTED_STATUSES = {
    "submitted",
    "scheduled",
    "queued",
    "running",
    "complete",
    "failed",
    "canceled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dynamic_repacking", "fixed_order"), required=True)
    parser.add_argument("--global-sequences", type=int, choices=GLOBAL_BATCHES, required=True)
    parser.add_argument("--target-epoch", type=int, choices=TARGETS, required=True)
    parser.add_argument("--learning-rate", default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help="Exact predecessor checkpoint, or the literal 'fresh' for an E1 bootstrap.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.target_epoch == 1 and args.source_checkpoint != "fresh":
        parser.error("E1 bootstrap requires --source-checkpoint=fresh")
    if args.target_epoch > 1 and args.source_checkpoint == "fresh":
        parser.error("E2+ requires an exact predecessor checkpoint")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", args.weight_decay):
        parser.error("--weight-decay must be a non-negative decimal")
    try:
        learning_rate = Decimal(args.learning_rate)
    except InvalidOperation:
        parser.error("--learning-rate must be a positive decimal or scientific-notation value")
    if not learning_rate.is_finite() or learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    wd = Decimal(args.weight_decay)
    lr2e3_exception = (
        args.method == "dynamic_repacking"
        and args.global_sequences == 256
        and learning_rate == Decimal("2e-3")
        and wd == Decimal("0.333")
    )
    if learning_rate > Decimal("2e-3") or (
        learning_rate == Decimal("2e-3") and not lr2e3_exception
    ):
        parser.error(
            "LR2e-3 is allowed only for the explicitly approved "
            "BS256 DR LR2e-3/WD0.333 trajectory; larger LR is prohibited"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    total_gpus = nodes_for(args.global_sequences) * GPUS_PER_NODE
    if args.global_sequences % (total_gpus * RANK_MICROBATCH_SEQUENCES):
        parser.error(
            f"global batch must be divisible by {total_gpus} ranks x rank microbatch eight; "
            f"got BS{args.global_sequences}"
        )
    return args


def numeric(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid numeric value {value!r}") from exc


def total_step(epoch: int, global_sequences: int) -> int:
    return math.ceil(epoch * TOKENS_PER_EPOCH / (global_sequences * SEQUENCE_LENGTH))


def stable_step(epoch: int, global_sequences: int) -> int:
    end = total_step(epoch, global_sequences)
    return end - round(DECAY_FRACTION * end) - 1


def warmup_steps(global_sequences: int) -> int:
    return 24_576 // global_sequences


def method_key(method: str, global_sequences: int) -> str:
    prefix = "dr" if method == "dynamic_repacking" else "fixed"
    return f"{prefix}{global_sequences}"


def canonical_number(value: str) -> str:
    number = numeric(value)
    if number == number.to_integral():
        return f"{number:.1f}"
    if abs(number) < Decimal("0.01"):
        exponent = number.adjusted()
        return f"{number.scaleb(-exponent).normalize()}e{exponent}"
    return str(number.normalize())


def trajectory_output(args: argparse.Namespace) -> str:
    mode = "_dr" if args.method == "dynamic_repacking" else ""
    return (
        f"{MODEL_ROOT}/bs{args.global_sequences}{mode}_"
        f"lr{canonical_number(args.learning_rate)}_wd{canonical_number(args.weight_decay)}"
    )


def nodes_for(global_sequences: int) -> int:
    # New BS64/128/256 work uses one 8-GPU node and scales gradient accumulation
    # to 1/2/4. Preserve the established four-node topology for BS512/1024.
    return 1 if global_sequences <= 256 else DEFAULT_NODES


def load_report() -> dict[str, Any]:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"run from the repository root; missing {REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text())


def active_epoch(run: dict[str, Any]) -> int:
    value = run.get("activeEpoch")
    return -1 if value is None else int(value)


def matching_registered_run(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    key = method_key(args.method, args.global_sequences)
    lr = numeric(args.learning_rate)
    wd = numeric(args.weight_decay)
    matches = [
        run
        for run in report.get("runs", [])
        if run.get("method") == key
        and int(run.get("batchSequences", 0)) == args.global_sequences
        and numeric(run.get("lr")) == lr
        and numeric(run.get("wd")) == wd
        and active_epoch(run) == args.target_epoch
    ]
    if len(matches) != 1:
        raise SystemExit(
            "refusing unregistered or ambiguous data-loader tuple: expected exactly one "
            f"{key}, LR={args.learning_rate}, WD={args.weight_decay}, "
            f"E{args.target_epoch} record; found {len(matches)}"
        )
    run = matches[0]
    if str(run.get("status", "")).lower() != "planned":
        raise SystemExit(
            f"refusing tuple in status {run.get('status')!r}; only a planned tuple may submit"
        )
    if str(args.target_epoch) in run.get("results", {}):
        raise SystemExit("refusing duplicate endpoint already present in results")
    if run.get("sourceExperiment") != args.source_experiment:
        raise SystemExit("source experiment does not match the registered exact predecessor")
    if run.get("sourceCheckpoint") != args.source_checkpoint:
        raise SystemExit("source checkpoint does not match the registered exact predecessor")
    nodes = nodes_for(args.global_sequences)
    if int(run.get("gpuCount", 0)) != nodes * GPUS_PER_NODE or int(run.get("nodeCount", 0)) != nodes:
        raise SystemExit(f"registered topology is not exactly {nodes} 8-GPU node(s)")
    if int(run.get("rankMicrobatchSequences", 0)) != RANK_MICROBATCH_SEQUENCES:
        raise SystemExit("registered rank microbatch is not eight sequences")

    for other in report.get("runs", []):
        if other is run or other.get("method") != key:
            continue
        if int(other.get("batchSequences", 0)) != args.global_sequences:
            continue
        if numeric(other.get("lr")) != lr:
            continue
        if numeric(other.get("wd")) != wd:
            continue
        result = other.get("results", {}).get(str(args.target_epoch))
        if result is not None or (
            active_epoch(other) == args.target_epoch
            and str(other.get("status", "")).lower() in ATTEMPTED_STATUSES
        ):
            raise SystemExit("refusing a duplicate attempted method/batch/WD/epoch tuple")

    if args.target_epoch > 1:
        predecessor = PREDECESSOR[args.target_epoch]
        expected_step = stable_step(predecessor, args.global_sequences)
        if not args.source_checkpoint.endswith(f"/step{expected_step}"):
            raise SystemExit(
                f"exact predecessor for E{args.target_epoch} must end in /step{expected_step}"
            )
    if args.target_epoch > 2:
        previous = run.get("results", {}).get(str(predecessor))
        if not isinstance(previous, dict) or previous.get("status") != "complete":
            raise SystemExit(
                "later endpoint requires a healthy completed same-coordinate predecessor"
            )
        previous_checkpoint = previous.get("preDecayCheckpoint")
        canonical_previous = f"{trajectory_output(args)}/step{expected_step}"
        if args.source_checkpoint not in {previous_checkpoint, canonical_previous}:
            raise SystemExit(
                "later endpoint source is neither its historical registered pre-decay "
                "checkpoint nor that exact step in the canonical trajectory directory"
            )

        # Each approved LR/WD trajectory now advances independently until its own
        # first validation non-improvement. Exact-source matching above guarantees
        # that a continuation keeps the same WD; a winner selected at another WD
        # must not prune this trajectory prematurely.
    return run


def beaker_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def extract_training_command(task: dict[str, Any]) -> tuple[str, str, list[str]]:
    arguments = task.get("arguments", [])
    if arguments[:1] == ["python"] and len(arguments) >= 3:
        return arguments[1], arguments[2], arguments[3:]
    if arguments[:2] == ["bash", "-lc"] and len(arguments) == 3:
        for line in arguments[2].splitlines():
            line = line.lstrip()
            if not line.startswith("torchrun "):
                continue
            parts = shlex.split(line)
            script_index = next(
                (index for index, part in enumerate(parts[1:], 1) if part.endswith(".py")),
                None,
            )
            if script_index is None or script_index + 1 >= len(parts):
                continue
            training_args = parts[script_index + 2 :]
            for index, value in enumerate(training_args):
                if value in {"|", "||", "&&", ";"}:
                    training_args = training_args[:index]
                    break
            return parts[script_index], parts[script_index + 1], training_args
    raise ValueError("expected a direct Python task or a bash task containing torchrun")


def values_for(arguments: list[str], prefix: str) -> list[str]:
    return [item.split("=", 1)[1] for item in arguments if item.startswith(prefix)]


def unique_value(arguments: list[str], prefix: str) -> str:
    values = values_for(arguments, prefix)
    if len(values) != 1:
        raise SystemExit(f"source experiment has {len(values)} values for {prefix}")
    return values[0]


def upsert(arguments: list[str], prefix: str, value: str) -> list[str]:
    return [item for item in arguments if not item.startswith(prefix)] + [value]


def audit_source_spec(spec: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    tasks = spec.get("tasks", [])
    nodes = nodes_for(args.global_sequences)
    if len(tasks) not in {1, nodes}:
        raise SystemExit(
            f"source experiment must contain one replicated task or exactly {nodes} "
            "materialized replica task(s)"
        )
    extracted = [extract_training_command(task) for task in tasks]
    if len(tasks) == nodes:
        if any(
            int(task.get("resources", {}).get("gpuCount", 0)) != GPUS_PER_NODE
            for task in tasks
        ):
            raise SystemExit(f"materialized source replicas are not {nodes} 8-GPU task(s)")
        if any(item != extracted[0] for item in extracted[1:]):
            raise SystemExit("materialized source replica training commands do not match")
        # Beaker expands a completed replicated task into one task per replica when its
        # spec is fetched. Canonicalize that representation before cloning the next job;
        # build_spec() will restore the intended four-replica declaration.
        spec["tasks"] = [copy.deepcopy(tasks[0])]
    script, _, arguments = extracted[0]
    if not script.endswith("olmo2-1B.py"):
        raise SystemExit(f"source uses unexpected training script {script!r}")
    if numeric(unique_value(arguments, "--lr=")) != numeric(args.learning_rate):
        raise SystemExit("source LR does not exactly match the requested coordinate")
    if args.target_epoch > 1 and numeric(
        unique_value(arguments, "--train_module.optim.weight_decay=")
    ) != numeric(args.weight_decay):
        raise SystemExit("source WD does not exactly match the requested coordinate")
    expected_batch = args.global_sequences * SEQUENCE_LENGTH
    source_batches = values_for(arguments, "--data_loader.global_batch_size=")
    if source_batches:
        if len(source_batches) != 1 or int(source_batches[0]) != expected_batch:
            raise SystemExit("source global batch does not match")
    elif args.global_sequences != 1024:
        # The earliest BS1024 WD sweep used the training script's historical default and did
        # not spell the global batch out as an override. All later sources must be explicit.
        raise SystemExit("non-BS1024 source is missing an explicit global batch")
    if args.target_epoch > 1:
        expected_step = stable_step(PREDECESSOR[args.target_epoch], args.global_sequences)
        if not args.source_checkpoint.endswith(f"/step{expected_step}"):
            raise SystemExit("source checkpoint is not the exact predecessor step")
        source_output = unique_value(arguments, "--save-folder=")
        historical_source = f"{source_output}/step{expected_step}"
        canonical_source = f"{trajectory_output(args)}/step{expected_step}"
        if args.source_checkpoint not in {historical_source, canonical_source}:
            raise SystemExit(
                "source checkpoint matches neither the source experiment output nor the "
                "canonical trajectory directory"
            )
    return script, arguments


def build_spec(
    base: dict[str, Any], args: argparse.Namespace, script: str, base_args: list[str]
) -> tuple[dict[str, Any], str]:
    nodes = nodes_for(args.global_sequences)
    spec = copy.deepcopy(base)
    task = spec["tasks"][0]
    mode_tag = "dr" if args.method == "dynamic_repacking" else "fixed"
    lr_tag = args.learning_rate.replace(".", "p")
    wd_tag = args.weight_decay.replace(".", "p")
    run_name = (
        f"dense_1b_step1_0802_repeated_dclm1b_wsd_bs{args.global_sequences}_"
        f"{mode_tag}_e{args.target_epoch}_lr{args.learning_rate}_wd{args.weight_decay}_"
        f"warmup{warmup_steps(args.global_sequences)}_{args.suffix}"
    )
    output = trajectory_output(args)
    train_args = [
        value
        for value in base_args
        if value not in {"--dynamic-repacking", "--fixed-data-order", "--no-data-shuffle"}
        and not value.startswith("--trainer.callbacks.checkpointer.save_interval=")
        and not value.startswith("--trainer.callbacks.checkpointer.ephemeral_save_interval=")
        and not value.startswith("--trainer.callbacks.downstream_evaluator.")
    ]
    if args.target_epoch > 1:
        train_args.append(
            "--dynamic-repacking"
            if args.method == "dynamic_repacking"
            else "--fixed-data-order"
        )
    else:
        # E1 is intentionally common to all data-order modes. Dynamic repacking or
        # fixed ordering begins only after this exact pre-decay checkpoint exists.
        train_args = [
            value
            for value in train_args
            if not value.startswith("--trainer.load_path=")
            and not value.startswith("--trainer.load_trainer_state=")
            and not value.startswith("--trainer.load_optim_state=")
            and not value.startswith("--trainer.reset_data_loader_state_on_load_path=")
            and not value.startswith(
                "--train_module.validate_optimizer_hyperparameters_on_load="
            )
        ]
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--dataset.mix=", "--dataset.mix=null"),
        ("--dataset.subset_manifest=", f"--dataset.subset_manifest={TRAIN_MANIFEST}"),
        (
            "--trainer.max_duration=",
            f"--trainer.max_duration={{value: {args.target_epoch * TOKENS_PER_EPOCH}, unit: tokens}}",
        ),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={run_name}"),
        (
            "--trainer.callbacks.wandb.tags=",
            (
                "--trainer.callbacks.wandb.tags="
                f"[pretraining,step1,data-loader-study,{mode_tag},bs{args.global_sequences},"
                f"e{args.target_epoch},lr{lr_tag},wd{wd_tag},wsd]"
            ),
        ),
        (
            "--trainer.callbacks.checkpointer.fixed_steps=",
            "--trainer.callbacks.checkpointer.fixed_steps="
            + json.dumps([stable_step(args.target_epoch, args.global_sequences)]),
        ),
        (
            "--data_loader.global_batch_size=",
            f"--data_loader.global_batch_size={args.global_sequences * SEQUENCE_LENGTH}",
        ),
        ("--data_loader.seed=", "--data_loader.seed=0"),
        (
            "--data_loader.restore_data_order_from_state=",
            "--data_loader.restore_data_order_from_state=false",
        ),
        (
            "--data_loader.ignore_fingerprint_mismatch=",
            "--data_loader.ignore_fingerprint_mismatch="
            + ("true" if args.method == "dynamic_repacking" else "false"),
        ),
        (
            "--train_module.rank_microbatch_size=",
            f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.scheduler=",
            (
                "--train_module.scheduler="
                f"{{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, "
                f"warmup: {warmup_steps(args.global_sequences)}, "
                f"decay_fraction: {DECAY_FRACTION}}}"
            ),
        ),
        (
            "--train_module.optim.weight_decay=",
            f"--train_module.optim.weight_decay={args.weight_decay}",
        ),
        ("--lr=", f"--lr={args.learning_rate}"),
    )
    if args.target_epoch > 1:
        replacements += (
            ("--force_exact_trainer_load_path=", "--force_exact_trainer_load_path=true"),
            ("--trainer.load_path=", f"--trainer.load_path={args.source_checkpoint}"),
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
    for prefix, value in replacements:
        train_args = upsert(train_args, prefix, value)
    train_args += [
        # Frontier decisions use DCLM held-out validation only. Skipping downstream
        # evaluation keeps completion ingestion and continuation launch lightweight.
        "--trainer.callbacks.downstream_evaluator.tasks=[]",
        "--trainer.callbacks.downstream_evaluator.eval_interval=null",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
    ]

    manifest_audit = (
        "import hashlib,json,pathlib\n"
        f"m=json.load(open({TRAIN_MANIFEST!r}))\n"
        "root=pathlib.Path('/weka/oe-training-default/ai2-llm')\n"
        "meta=(root/m['materialized']['document_metadata_path']).resolve()\n"
        "assert meta.is_file(), meta\n"
        "h=hashlib.sha256()\n"
        "with meta.open('rb') as f:\n"
        "  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)\n"
        "assert h.hexdigest()==m['materialized']['document_metadata_sha256']\n"
        "assert len(m['entries'])==1 and m['entries'][0]['start_instance']==0\n"
        "print('DATA_LOADER_MANIFEST_OK',meta)"
    )
    preflight_steps = []
    target_pre_decay = f"{output}/step{stable_step(args.target_epoch, args.global_sequences)}"
    target_endpoint = f"{output}/step{total_step(args.target_epoch, args.global_sequences)}"
    if args.target_epoch > 1:
        # E2 starts a new canonical DR trajectory, so its output directory is
        # intentionally absent until the trainer creates it. Later frontiers
        # must reuse the already-established canonical trajectory directory.
        if args.target_epoch > 2:
            preflight_steps.append(shlex.join(["test", "-d", output]))
        else:
            preflight_steps.append(shlex.join(["test", "-d", str(MODEL_ROOT)]))
        preflight_steps.append(shlex.join(["test", "-d", args.source_checkpoint]))
    else:
        preflight_steps.append(shlex.join(["test", "!", "-e", output]))
    preflight_steps.extend(
        [
            shlex.join(["test", "-f", TRAIN_MANIFEST]),
            shlex.join(["test", "!", "-e", target_pre_decay]),
            shlex.join(["test", "!", "-e", target_endpoint]),
            shlex.join(["python", "-c", manifest_audit]),
            shlex.join(
                [
                    "echo",
                    (
                        f"DATA_LOADER_PREFLIGHT_OK method={mode_tag} "
                        f"bs={args.global_sequences} epoch={args.target_epoch} "
                        f"lr={args.learning_rate} wd={args.weight_decay} "
                        f"source={args.source_checkpoint} nodes={nodes} rank_mb=8"
                    ),
                ]
            ),
            shlex.join(["python", script, run_name, "--dry-run", *train_args]),
        ]
    )
    preflight = "\n".join(preflight_steps)
    rendezvous = hashlib.sha256(run_name.encode()).hexdigest()
    launch = (
        'torchrun --nnodes="$BEAKER_REPLICA_COUNT:$BEAKER_REPLICA_COUNT" '
        '--nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
        '--rdzv-id="$GANTRY_RDZV_ID" --rdzv-backend=static '
        '--rdzv-endpoint="$BEAKER_LEADER_REPLICA_HOSTNAME:$GANTRY_RDZV_PORT" '
        '--node-rank="$BEAKER_REPLICA_RANK" --rdzv-conf="read_timeout=420" '
        + shlex.join([script, run_name, *train_args])
    )
    task["arguments"] = [
        "bash",
        "-lc",
        (
            "set -euo pipefail\n"
            'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then\n'
            f"{preflight}\n"
            "fi\n"
            f"{launch}"
        ),
    ]
    blocked_env = {
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "GANTRY_USE_TORCHRUN",
        "NUM_NODES",
    }
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in blocked_env
        and not (
            str(env.get("name", "")).startswith("BEAKER_") and env.get("name") != "BEAKER_TOKEN"
        )
    ]
    for env in task["envVars"]:
        if env.get("name") == "GIT_REF":
            env["value"] = args.revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": args.revision})
    task["envVars"] += [
        {"name": "NUM_NODES", "value": str(nodes)},
        {"name": "GANTRY_RDZV_ID", "value": rendezvous[:12]},
        {
            "name": "GANTRY_RDZV_PORT",
            "value": str(29_000 + int(rendezvous[:8], 16) % 1_000),
        },
    ]
    task["resources"] = {"gpuCount": GPUS_PER_NODE, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": args.priority,
        "minRuntime": "0s",
        "autoResume": False,
    }
    task["replicas"] = nodes
    task["leaderSelection"] = True
    task["hostNetworking"] = True
    task["propagateFailure"] = True
    task["propagatePreemption"] = True
    if nodes > 1:
        task["synchronizedStartTimeout"] = "90m"
    else:
        task.pop("synchronizedStartTimeout", None)
    spec.pop("description", None)
    return spec, output


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def register_submission(
    report: dict[str, Any], run: dict[str, Any], experiment: str, output: str, revision: str
) -> None:
    if run.get("status") != "planned":
        raise RuntimeError("registry changed after submission; refusing to overwrite it")
    run["status"] = "submitted"
    run["beaker"] = experiment
    run["experiment"] = experiment
    run["revision"] = revision
    run["output"] = output
    run["reason"] = (
        "Submitted locally after exact-source, canonical trajectory-output, LR/WD, "
        "topology, rank-microbatch, manifest-checksum, and duplicate-tuple guards passed."
    )
    report["updated"] = "2026-08-13"
    write_report(report)


def main() -> None:
    args = parse_args()
    report = load_report()
    run = matching_registered_run(report, args)
    base = beaker_spec(args.source_experiment)
    script, base_args = audit_source_spec(base, args)
    spec, output = build_spec(base, args, script, base_args)
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
        ],
        check=True,
        input=json.dumps(spec),
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")
    if args.register:
        ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", result.stdout)
        if not ids:
            raise RuntimeError("submission succeeded without a parsed experiment ID")
        register_submission(report, run, ids[0], output, args.revision)


if __name__ == "__main__":
    main()
