#!/usr/bin/env python3
"""Submit one guarded nested-3B continuation.

E1 restores the selected 1B-pool pre-decay model/optimizer state, resets the
loader, and consumes only the disjoint 2B extension. Later frontiers restore
the selected preceding 3B-pool checkpoint, reset/reshuffle the loader, and
repeat the complete nested 3B pool. LR is fixed per model/batch throughout.
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
from decimal import Decimal
from pathlib import Path
from typing import Any


SEQ = 4096
POOL_TOKENS = 3_000_000_000
DECAY = 0.1
ROOT = "/weka/oe-training-default/sewonm/icsl/data/dclm_0802_nested_1b_3b_9b"
EXTENSION = f"{ROOT}/manifests/dclm_0802_nested_extension_1b_to_3b.json"
FULL_POOL = f"{ROOT}/manifests/dclm_0802_nested_train_3b.json"
POOL_MANIFEST = f"{ROOT}/manifests/dclm_0802_nested_1b_3b_9b.pool.json"
TARGETS = (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
PREDECESSOR = dict(zip(TARGETS[1:], TARGETS[:-1]))
WD_LADDER = tuple(Decimal(x) for x in ("0.01", "0.033", "0.1", "0.3", "0.333", "1.0"))
MODELS = {
    "153m": {
        "lr": {64: "2e-3", 128: "2e-3", 256: "2e-3"},
        "rank_mb": 16,
        "size": "153M",
        "report": "wsd_batch_size_153m_pool3b.json",
    },
    "474m": {
        "lr": {64: "2e-3", 128: "2e-3", 256: "2e-3"},
        "rank_mb": 16,
        "size": "474M",
        "report": "wsd_batch_size_474m_pool3b.json",
    },
    "1b": {
        "lr": {64: "1e-3", 128: "1e-3", 256: "1e-3"},
        "rank_mb": 8,
        "size": "1B",
        "report": "wsd_batch_size_1b_pool3b.json",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--global-sequences", choices=(64, 128, 256), type=int, required=True)
    p.add_argument("--target-epoch", choices=TARGETS, type=int, default=1)
    p.add_argument("--weight-decay", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--base-experiment", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--suffix", required=True)
    p.add_argument("--workspace", default="ai2/flex2")
    p.add_argument("--priority", default="urgent")
    p.add_argument("--recover-failed-experiment")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--register", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", a.suffix):
        p.error("suffix must be a lowercase run-name component")
    source_tokens = (
        1_000_000_000
        if a.target_epoch == 1
        else PREDECESSOR[a.target_epoch] * POOL_TOKENS
    )
    expected = stable_step(source_tokens, a.global_sequences)
    if not a.source_checkpoint.endswith(f"/step{expected}"):
        p.error(f"source must be the exact pre-decay checkpoint /step{expected}")
    return a


def lr_for(model: str, batch: int) -> str:
    return MODELS[model]["lr"][batch]


def topology(model: str, batch: int) -> tuple[int, int, int]:
    """Return GPUs per node, node count, and total GPUs at grad accumulation one."""
    rank_mb = MODELS[model]["rank_mb"]
    if batch % rank_mb:
        raise ValueError(f"BS{batch} is not divisible by rank microbatch {rank_mb}")
    total_gpus = batch // rank_mb
    if total_gpus <= 8:
        return total_gpus, 1, total_gpus
    if total_gpus % 8:
        raise ValueError(f"total GPU count {total_gpus} is not divisible by 8")
    return 8, total_gpus // 8, total_gpus


def total_steps(tokens: int, batch: int) -> int:
    return math.ceil(tokens / (batch * SEQ))


def stable_step(tokens: int, batch: int) -> int:
    end = total_steps(tokens, batch)
    return end - round(DECAY * end) - 1


def retained_steps(source_epoch: int, target_epoch: int, batch: int) -> list[int]:
    return [
        stable_step(epoch * POOL_TOKENS, batch)
        for epoch in range(source_epoch + 1, target_epoch + 1)
    ]


def upsert(args: list[str], prefix: str, value: str) -> list[str]:
    return [item for item in args if not item.startswith(prefix)] + [value]


def base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def report_path(model: str) -> Path:
    return Path("reports/0802/data") / MODELS[model]["report"]


def extract_training_command(task: dict[str, Any]) -> tuple[str, str, list[str]]:
    arguments = task.get("arguments", [])
    if arguments[:1] == ["python"] and len(arguments) >= 3:
        return arguments[1], arguments[2], arguments[3:]
    if arguments[:2] == ["bash", "-lc"] and len(arguments) == 3:
        for line in arguments[2].splitlines():
            parts = shlex.split(line)
            if len(parts) >= 4 and parts[0] == "torchrun":
                script_index = next(
                    (index for index, part in enumerate(parts[1:], 1) if part.endswith(".py")),
                    None,
                )
                if script_index is not None and script_index + 1 < len(parts):
                    return (
                        parts[script_index],
                        parts[script_index + 1],
                        parts[script_index + 2 :],
                    )
    raise ValueError("expected a direct Python task or a bash task containing torchrun")


def unique_arg(arguments: list[str], prefix: str) -> str:
    values = [item.split("=", 1)[1] for item in arguments if item.startswith(prefix)]
    if len(values) != 1:
        raise SystemExit(f"Beaker provenance has {len(values)} values for {prefix}")
    return values[0]


def optional_arg(arguments: list[str], prefix: str) -> str | None:
    values = [item.split("=", 1)[1] for item in arguments if item.startswith(prefix)]
    if len(values) > 1:
        raise SystemExit(f"Beaker provenance has {len(values)} values for {prefix}")
    return values[0] if values else None


def audit_model_args(model: str, arguments: list[str]) -> None:
    output = unique_arg(arguments, "--save-folder=")
    if f"/dense_{model}_" not in output:
        raise SystemExit(f"Beaker output path does not identify dense_{model}")
    model_size = optional_arg(arguments, "--model-size=")
    if model == "153m":
        if model_size != "153M":
            raise SystemExit("153M Beaker predecessor has the wrong model-size")
    elif model == "474m":
        expected = {
            "--model-size=": "1B",
            "--model.d_model=": "1024",
            "--model.n_layers=": "16",
            "--model.block.sequence_mixer.n_heads=": "16",
            "--model.block.feed_forward.hidden_size=": "4096",
        }
        actual = {prefix: optional_arg(arguments, prefix) for prefix in expected}
        if actual != expected:
            raise SystemExit(f"474M Beaker architecture mismatch: {actual}")
    elif model == "1b":
        if model_size not in {None, "1B"}:
            raise SystemExit("1B Beaker predecessor has the wrong model-size")
        for prefix in ("--model.d_model=", "--model.n_layers="):
            if optional_arg(arguments, prefix) is not None:
                raise SystemExit(f"1B Beaker predecessor unexpectedly overrides {prefix}")


def result_checkpoint(result: dict[str, Any], batch: int) -> str | None:
    checkpoint = result.get("resumeCheckpoint")
    if checkpoint:
        return checkpoint
    output = result.get("output")
    if output:
        return f"{output}/step{stable_step(1_000_000_000, batch)}"
    return None


def unhealthy_ids(report: dict[str, Any], baseline: bool = False) -> set[str]:
    health = (
        report.get("baseline1b", {}).get("healthAudit", {})
        if baseline
        else report.get("healthAudit", {})
    )
    return set(health.get("unhealthy", {}))


def completed_candidates(
    sweeps: list[dict[str, Any]], batch: int, epoch: int, lr: str,
    unhealthy: set[str],
) -> list[tuple[Decimal, Decimal, dict[str, Any], dict[str, Any]]]:
    candidates = []
    for sweep in sweeps:
        if (
            sweep.get("batchSequences") != batch
            or Decimal(str(sweep.get("lr"))) != Decimal(lr)
        ):
            continue
        result = sweep.get("results", {}).get(str(epoch))
        if (
            not isinstance(result, dict)
            or result.get("status") != "complete"
            or result.get("validation") is None
            or result.get("wandb") in unhealthy
        ):
            continue
        candidates.append(
            (
                Decimal(str(result["validation"])),
                Decimal(str(sweep["wd"])),
                sweep,
                result,
            )
        )
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def selected_3b(report: dict[str, Any], batch: int, epoch: int, lr: str):
    candidates = completed_candidates(
        report.get("batchSweeps", []), batch, epoch, lr, unhealthy_ids(report)
    )
    if not candidates:
        raise SystemExit(f"E{epoch} is not resolved for BS{batch} at fixed LR {lr}")
    return candidates[0]


def baseline_wd_cap(report: dict[str, Any], batch: int, epoch: int, lr: str) -> Decimal:
    candidates = completed_candidates(
        report.get("baseline1b", {}).get("batchSweeps", []),
        batch,
        epoch,
        lr,
        unhealthy_ids(report, baseline=True),
    )
    if not candidates:
        raise SystemExit(
            f"the embedded 1B-pool report has no trusted BS{batch} E{epoch} result "
            f"at fixed LR {lr}; refusing to invent a WD cap"
        )
    return candidates[0][1]


def baseline_horizon(report: dict[str, Any], batch: int, lr: str) -> int:
    """Return the first configured target at/after the best trusted 1B epoch."""
    sweeps = report.get("baseline1b", {}).get("batchSweeps", [])
    bad = unhealthy_ids(report, baseline=True)
    rows = []
    epochs = {
        int(epoch)
        for sweep in sweeps
        if sweep.get("batchSequences") == batch
        for epoch in sweep.get("results", {})
        if str(epoch).isdigit() and int(epoch) >= 1
    }
    for epoch in epochs:
        candidates = completed_candidates(sweeps, batch, epoch, lr, bad)
        if candidates:
            rows.append((candidates[0][0], epoch))
    if not rows:
        raise SystemExit(f"no trusted 1B-pool horizon exists for BS{batch} at LR {lr}")
    optimum = min(rows)[1]
    return next((target for target in TARGETS if target >= optimum), TARGETS[-1])


def audit_later(
    a: argparse.Namespace, report: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    batch = a.global_sequences
    lr = lr_for(a.model, batch)
    predecessor_epoch = PREDECESSOR[a.target_epoch]
    selected_ce, selected_wd, selected_sweep, selected_result = selected_3b(
        report, batch, predecessor_epoch, lr
    )
    if selected_sweep.get("beaker") != a.base_experiment:
        raise SystemExit(
            "base experiment must be the selected preceding 3B-pool frontier"
        )
    source_output = selected_result.get("output") or selected_sweep.get("output")
    expected_checkpoint = (
        f"{source_output}/step{stable_step(predecessor_epoch * POOL_TOKENS, batch)}"
    )
    if a.source_checkpoint != expected_checkpoint:
        raise SystemExit(
            f"wrong selected predecessor checkpoint: expected {expected_checkpoint}"
        )
    requested_wd = Decimal(a.weight_decay)
    if selected_wd not in WD_LADDER:
        raise SystemExit(f"selected WD {selected_wd} is not on the configured ladder")
    selected_index = WD_LADDER.index(selected_wd)
    allowed = {selected_wd}
    if selected_index + 1 < len(WD_LADDER):
        allowed.add(WD_LADDER[selected_index + 1])
    if requested_wd not in allowed:
        raise SystemExit(
            f"E{a.target_epoch} WD must be selected predecessor WD {selected_wd} "
            "or exactly one ladder step higher"
        )
    cap = baseline_wd_cap(report, batch, a.target_epoch, lr)
    if requested_wd > cap:
        raise SystemExit(
            f"WD {requested_wd} exceeds the trusted 1B-pool BS{batch} "
            f"E{a.target_epoch} cap {cap}"
        )
    horizon = baseline_horizon(report, batch, lr)
    if a.target_epoch > horizon:
        raise SystemExit(
            f"E{a.target_epoch} exceeds the 1B-pool stopping horizon E{horizon}"
        )
    earlier_epoch = PREDECESSOR.get(predecessor_epoch)
    if earlier_epoch is not None:
        earlier_ce, _, _, _ = selected_3b(report, batch, earlier_epoch, lr)
        if selected_ce >= earlier_ce:
            raise SystemExit(
                f"strict stop: selected E{predecessor_epoch} validation {selected_ce} "
                f"did not improve on E{earlier_epoch} validation {earlier_ce}"
            )
    duplicates = [
        sweep for sweep in report.get("batchSweeps", [])
        if sweep.get("batchSequences") == batch
        and Decimal(str(sweep.get("lr"))) == Decimal(lr)
        and Decimal(str(sweep.get("wd"))) == requested_wd
        and sweep.get("activeEpoch") == a.target_epoch
    ]
    if duplicates:
        live = [s for s in duplicates if s.get("status") != "failed"]
        recovered = [
            s for s in duplicates
            if s.get("status") == "failed"
            and s.get("beaker") == a.recover_failed_experiment
        ]
        if live or len(recovered) != 1:
            raise SystemExit("refusing duplicate registered later-frontier tuple")
    elif a.recover_failed_experiment:
        raise SystemExit("recovery experiment does not match a registered failed tuple")
    _, _, source_args = extract_training_command(source["tasks"][0])
    if (
        Decimal(unique_arg(source_args, "--lr=")) != Decimal(lr)
        or Decimal(unique_arg(source_args, "--train_module.optim.weight_decay="))
        != selected_wd
        or unique_arg(source_args, "--save-folder=") != source_output
    ):
        raise SystemExit("selected predecessor disagrees with its Beaker specification")
    audit_model_args(a.model, source_args)
    return selected_sweep


def audit(
    a: argparse.Namespace, report: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    if a.target_epoch != 1:
        return audit_later(a, report, source)
    plan = report["poolPlan"]
    batch = str(a.global_sequences)
    lr = lr_for(a.model, a.global_sequences)
    if Decimal(plan["fixedLearningRateByBatch"][batch]) != Decimal(lr):
        raise SystemExit("report LR policy does not match launcher")
    candidates = plan["initialWeightDecayByBatch"][batch]
    allowed = {Decimal(x) for x in candidates}
    if Decimal(a.weight_decay) not in allowed:
        raise SystemExit(f"WD {a.weight_decay} is outside initial candidates {sorted(allowed)}")
    registered = [
        sweep
        for sweep in report["baseline1b"]["batchSweeps"]
        if sweep.get("beaker") == a.base_experiment
    ]
    if len(registered) != 1:
        raise SystemExit(
            "source experiment must occur exactly once in baseline1b.batchSweeps"
        )
    predecessor = registered[0]
    result = predecessor.get("results", {}).get("1", {})
    source_wd = Decimal(candidates[0])
    report_tuple = (
        predecessor.get("batchSequences"),
        Decimal(str(predecessor.get("lr"))),
        Decimal(str(predecessor.get("wd"))),
        result_checkpoint(result, a.global_sequences),
    )
    source_tuple = (
        a.global_sequences,
        Decimal(lr),
        source_wd,
        a.source_checkpoint,
    )
    if report_tuple != source_tuple or result.get("status") != "complete":
        raise SystemExit(
            f"report predecessor mismatch: requested={source_tuple}, report={report_tuple}"
        )
    _, _, base_arguments = extract_training_command(source["tasks"][0])
    beaker_tuple = (
        int(unique_arg(base_arguments, "--data_loader.global_batch_size=")) // SEQ,
        Decimal(unique_arg(base_arguments, "--lr=")),
        Decimal(unique_arg(base_arguments, "--train_module.optim.weight_decay=")),
        unique_arg(base_arguments, "--save-folder=")
        + f"/step{stable_step(1_000_000_000, a.global_sequences)}",
    )
    if beaker_tuple != source_tuple:
        raise SystemExit(
            f"Beaker predecessor mismatch: requested={source_tuple}, beaker={beaker_tuple}"
        )
    audit_model_args(a.model, base_arguments)
    duplicates = [
        sweep
        for sweep in report.get("batchSweeps", [])
        if (
            sweep.get("batchSequences") == a.global_sequences
            and Decimal(str(sweep.get("lr"))) == Decimal(lr)
            and Decimal(str(sweep.get("wd"))) == Decimal(a.weight_decay)
            and sweep.get("activeEpoch") == a.target_epoch
        )
    ]
    if duplicates:
        live = [s for s in duplicates if s.get("status") != "failed"]
        recovered = [
            s for s in duplicates
            if s.get("status") == "failed"
            and s.get("beaker") == a.recover_failed_experiment
        ]
        if live or len(recovered) != 1:
            raise SystemExit("refusing duplicate registered E1 tuple")
    elif a.recover_failed_experiment:
        raise SystemExit("recovery experiment does not match a registered failed tuple")
    return predecessor


def build(
    a: argparse.Namespace, source: dict[str, Any], predecessor: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    spec = copy.deepcopy(source)
    task = spec["tasks"][0]
    # ``beaker experiment spec`` materializes one task per replica for an
    # already-submitted multi-node predecessor. Reconstruct one clean logical
    # task before applying the requested replica topology; otherwise a later
    # frontier would accidentally retain stale replica tasks and env values.
    spec["tasks"] = [task]
    task["name"] = "main"
    script, _, args = extract_training_command(task)
    blocked = (
        "--load_path=", "--load_trainer_state=", "--trainer.load_path=",
        "--trainer.load_trainer_state=", "--trainer.load_optim_state=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--trainer.callbacks.checkpointer.save_interval=",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
    )
    args = [item for item in args if not item.startswith(blocked)]
    lr = lr_for(a.model, a.global_sequences)
    warmup = int(predecessor["warmupSteps"])
    source_epoch = 0 if a.target_epoch == 1 else PREDECESSOR[a.target_epoch]
    data_manifest = EXTENSION if a.target_epoch == 1 else FULL_POOL
    name = (
        f"dense_{a.model}_step1_pool3b_nested_bs{a.global_sequences}_e{a.target_epoch}_"
        f"lr{lr}_wd{a.weight_decay}_warmup{warmup}_{a.suffix}"
    )
    output = f"/weka/oe-training-default/sewonm/icsl/models/{name}"
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--dataset.subset_manifest=", f"--dataset.subset_manifest={data_manifest}"),
        ("--dataset.mix=", "--dataset.mix=null"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {a.target_epoch * POOL_TOKENS}, unit: tokens}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        ("--trainer.callbacks.wandb.tags=", "--trainer.callbacks.wandb.tags=" +
         f"[pretraining,step1,nested-3b,loader-reset,dense-{a.model},bs{a.global_sequences},e{a.target_epoch},wsd]"),
        ("--trainer.callbacks.checkpointer.fixed_steps=", "--trainer.callbacks.checkpointer.fixed_steps=" + json.dumps(retained_steps(source_epoch, a.target_epoch, a.global_sequences), separators=(",", ":"))),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={a.global_sequences * SEQ}"),
        ("--data_loader.seed=", f"--data_loader.seed={80_200 + a.target_epoch}"),
        ("--train_module.rank_microbatch_size=", f"--train_module.rank_microbatch_size={MODELS[a.model]['rank_mb'] * SEQ}"),
        ("--train_module.scheduler=", "--train_module.scheduler=" +
         f"{{_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: {warmup}, decay_fraction: {DECAY}}}"),
        ("--train_module.optim.weight_decay=", f"--train_module.optim.weight_decay={a.weight_decay}"),
        ("--lr=", f"--lr={lr}"),
        ("--trainer.load_path=", f"--trainer.load_path={a.source_checkpoint}"),
        ("--trainer.load_trainer_state=", "--trainer.load_trainer_state=true"),
        ("--trainer.load_optim_state=", "--trainer.load_optim_state=true"),
        ("--trainer.reset_data_loader_state_on_load_path=", "--trainer.reset_data_loader_state_on_load_path=true"),
    )
    for prefix, value in replacements:
        args = upsert(args, prefix, value)
    args += [
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
    ]
    checks = [
        "set -euo pipefail",
        shlex.join(["test", "-d", a.source_checkpoint]),
        shlex.join(["test", "-f", data_manifest]),
        shlex.join(["test", "-f", POOL_MANIFEST]),
        "python - <<'PY'\nimport json\np=json.load(open('" + POOL_MANIFEST + "'))\n"
        "assert p['audit']['passed'] is True\nassert p['audit']['base_document_overlap']==0\n"
        "assert p['audit']['chunk_document_overlap']==0\nprint('POOL3B_AUDIT_OK')\nPY",
        shlex.join(["test", "!", "-e", output]),
        shlex.join(["echo", f"POOL3B_PREFLIGHT model={a.model} bs={a.global_sequences} epoch={a.target_epoch} lr={lr} wd={a.weight_decay} source={a.source_checkpoint} manifest={data_manifest} reset_loader=true seed={80_200 + a.target_epoch}"]),
        shlex.join(["python", script, name, "--dry-run", *args]),
    ]
    gpus_per_node, nodes, _ = topology(a.model, a.global_sequences)
    preflight = "\n".join(checks[1:])
    if nodes == 1:
        launch = (
            'torchrun --nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
            + shlex.join([script, name, *args])
        )
    else:
        launch = (
            'torchrun '
            '--nnodes="$BEAKER_REPLICA_COUNT:$BEAKER_REPLICA_COUNT" '
            '--nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
            '--rdzv-id="$GANTRY_RDZV_ID" '
            '--rdzv-backend=static '
            '--rdzv-endpoint="$BEAKER_LEADER_REPLICA_HOSTNAME:$GANTRY_RDZV_PORT" '
            '--node-rank="$BEAKER_REPLICA_RANK" '
            '--rdzv-conf="read_timeout=420" '
            + shlex.join([script, name, *args])
        )
    task["arguments"] = [
        "bash",
        "-lc",
        (
            'set -euo pipefail\n'
            'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then\n'
            f'{preflight}\n'
            'fi\n'
            f'{launch}'
        ),
    ]
    blocked_env = {
        "GANTRY_POST_SETUP_CMD",
        "GANTRY_RDZV_ID",
        "GANTRY_RDZV_PORT",
        "GANTRY_USE_TORCHRUN",
        "NUM_NODES",
    }
    task["envVars"] = [
        x
        for x in task.get("envVars", [])
        if x.get("name") not in blocked_env
        and not (
            str(x.get("name", "")).startswith("BEAKER_")
            and x.get("name") != "BEAKER_TOKEN"
        )
    ]
    for env in task["envVars"]:
        if env.get("name") == "GIT_REF":
            env["value"] = a.revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": a.revision})
    task["envVars"] += [
        {"name": "NUM_NODES", "value": str(nodes)},
    ]
    if nodes > 1:
        rendezvous = hashlib.sha256(name.encode()).hexdigest()
        task["envVars"] += [
            {"name": "GANTRY_RDZV_ID", "value": rendezvous[:12]},
            {"name": "GANTRY_RDZV_PORT", "value": str(29_000 + int(rendezvous[:8], 16) % 1_000)},
        ]
    task["resources"] = {"gpuCount": gpus_per_node, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": a.priority, "minRuntime": "0s", "autoResume": False}
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
    return spec, name, output


def register(a: argparse.Namespace, experiment: str, output: str) -> None:
    path = report_path(a.model)
    report = json.loads(path.read_text())
    gpus_per_node, nodes, total_gpus = topology(a.model, a.global_sequences)
    rank_mb = MODELS[a.model]["rank_mb"]
    predecessor_pool = (
        report["baseline1b"]["batchSweeps"]
        if a.target_epoch == 1
        else report["batchSweeps"]
    )
    predecessors = [
        sweep for sweep in predecessor_pool if sweep.get("beaker") == a.base_experiment
    ]
    if len(predecessors) != 1:
        raise RuntimeError("registered source predecessor is no longer unique")
    report["batchSweeps"].append({
        "batchSequences": a.global_sequences,
        "globalBatchTokens": a.global_sequences * SEQ,
        "contextLength": SEQ,
        "lr": lr_for(a.model, a.global_sequences),
        "wd": a.weight_decay,
        "warmupSteps": int(predecessors[0]["warmupSteps"]),
        "rankMicrobatchSequences": rank_mb,
        "gradientAccumulation": 1,
        "gpuCountPerNode": gpus_per_node,
        "nodeCount": nodes,
        "gpuCount": total_gpus,
        "status": "pending",
        "activeEpoch": a.target_epoch,
        "search": "nested-3b-fixed-lr-adaptive-wd",
        "beaker": experiment,
        "output": output,
        "sourceCheckpoint": a.source_checkpoint,
        "sourceEpoch": 0 if a.target_epoch == 1 else PREDECESSOR[a.target_epoch],
        "dataManifest": EXTENSION if a.target_epoch == 1 else FULL_POOL,
        "dataLoaderReset": True,
        "dataLoaderSeed": 80_200 + a.target_epoch,
        "retainedPreDecaySteps": retained_steps(
            0 if a.target_epoch == 1 else PREDECESSOR[a.target_epoch],
            a.target_epoch,
            a.global_sequences,
        ),
        "actualTargetTokens": a.target_epoch * POOL_TOKENS,
        "revision": a.revision,
        "recoveryOf": a.recover_failed_experiment,
        "results": {},
    })
    report["updated"] = "2026-08-09"
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text("window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")


def main() -> None:
    a = parse_args()
    report = json.loads(report_path(a.model).read_text())
    source = base_spec(a.base_experiment)
    predecessor = audit(a, report, source)
    spec, name, output = build(a, source, predecessor)
    if a.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", a.workspace, "--priority", a.priority],
        check=True, input=json.dumps(spec), text=True, stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", result.stdout)
    if a.register:
        if not ids:
            raise RuntimeError("submission succeeded without a parsed experiment ID")
        register(a, ids[0], output)


if __name__ == "__main__":
    main()
