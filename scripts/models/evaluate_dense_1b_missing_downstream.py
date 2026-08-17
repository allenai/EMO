#!/usr/bin/env python3
"""Evaluate every retained Dense-1B data-loader endpoint missing downstream metrics.

The report's DR, DR+WT, and DR+WT+EmbedWD training jobs deliberately skipped
downstream evaluation. This launcher creates one independent Beaker experiment
per batch-size/method pair. Each experiment contains exactly one one-GPU task
which scans real ``*_dr_*`` Weka output directories and evaluates that pair's
missing epochs serially.

The same script refreshes a registered campaign and writes completed metrics back
to ``wsd_data_loader_1b.json`` plus its JavaScript mirror.

Examples:

    # Local, network-free inventory.
    .venv/bin/python scripts/models/evaluate_dense_1b_missing_downstream.py --plan

    # Render the Beaker spec without submitting it (reads source specs from Beaker).
    .venv/bin/python scripts/models/evaluate_dense_1b_missing_downstream.py \
        --print-spec --revision <pushed-revision>

    # Submit independent experiments containing one sequential 1-GPU task each.
    .venv/bin/python scripts/models/evaluate_dense_1b_missing_downstream.py \
        --submit --revision <pushed-revision>

    # Refresh task states and ingest all complete nine-task results.
    .venv/bin/python scripts/models/evaluate_dense_1b_missing_downstream.py --refresh
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
WORKSPACE = "ai2/flex2"
DEFAULT_NAME = "dense-1b-missing-downstream-evals"
CHECKPOINT_CAP = 128
SEQUENTIAL_MANIFEST = Path("scripts/models/manifests/dense_1b_downstream.json")
DOWNSTREAM_TASK_ARGUMENT = (
    "[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, "
    "openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]"
)
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
SCOPED_METHODS = {
    "dr64",
    "dr128",
    "dr256",
    "dr512",
    "dr1024",
    "drwt64",
    "drwt512",
    "drwtembwd64",
    "drwtembwd512",
}
ACTIVE = {"submitted", "scheduled", "running"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ACCURACY = re.compile(r"\s([a-z0-9_]+) \((?:length-normalized )?accuracy\)=([0-9.]+)")
BPB = re.compile(r"\s([a-z0-9_]+) \(BPB\)=([0-9.]+)")
WANDB_RUN = re.compile(r"https://wandb\.ai/[^\s]+/runs/([a-zA-Z0-9_-]+)")
ULID = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the local checkpoint inventory.")
    mode.add_argument("--print-spec", action="store_true", help="Build and print the Beaker spec.")
    mode.add_argument("--submit", action="store_true", help="Submit and register the campaign.")
    mode.add_argument(
        "--refresh", action="store_true", help="Refresh registered campaigns and ingest metrics."
    )
    parser.add_argument("--revision", help="Pushed revision used by evaluation tasks.")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--replace-deleted-experiment",
        help="Replace this exact deleted campaign registration after a successful submission.",
    )
    parser.add_argument(
        "--supersede-experiment",
        help=(
            "Ignore this failed campaign while rebuilding the missing inventory, then mark it "
            "superseded after all independent experiments submit successfully."
        ),
    )
    args = parser.parse_args()
    if (args.submit or args.print_spec) and not args.revision:
        parser.error("--revision is required with --submit or --print-spec")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must contain only lowercase letters, digits, and hyphens")
    return args


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def load_report(path: Path = REPORT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def has_full_downstream(result: dict[str, Any]) -> bool:
    accuracy = result.get("downstream") or {}
    bpb = result.get("downstreamBpb") or {}
    return all(task in accuracy and task in bpb for task in REPORT_TASKS)


def checkpoint_for(result: dict[str, Any]) -> str | None:
    return result.get("endpointCheckpoint") or result.get("checkpoint")


def variant(run_record: dict[str, Any]) -> str:
    if run_record.get("decayEmbeddings"):
        return "dr-wt-embwd"
    if run_record.get("weightTying"):
        return "dr-wt"
    return "dr"


def existing_campaign_checkpoints(
    report: dict[str, Any], *, ignored_experiments: set[str] | None = None
) -> set[str]:
    ignored_experiments = ignored_experiments or set()
    checkpoints: set[str] = set()
    for campaign in report.get("downstreamEvaluationCampaigns", []):
        if campaign.get("experiment") in ignored_experiments:
            continue
        for task in campaign.get("tasks", []):
            if task.get("status") in ACTIVE | {"complete"} and task.get("checkpoint"):
                checkpoints.add(task["checkpoint"])
    return checkpoints


def candidates(
    report: dict[str, Any], *, ignored_experiments: set[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    already_registered = existing_campaign_checkpoints(
        report, ignored_experiments=ignored_experiments
    )
    found: list[dict[str, Any]] = []
    no_checkpoint: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_record in report.get("runs", []):
        method = run_record.get("method")
        if method not in SCOPED_METHODS:
            continue
        for epoch_text, result in (run_record.get("results") or {}).items():
            if result.get("status") != "complete" or has_full_downstream(result):
                continue
            checkpoint = checkpoint_for(result)
            item = {
                "runId": run_record["id"],
                "method": method,
                "variant": variant(run_record),
                "batchSequences": int(run_record["batchSequences"]),
                "epoch": int(epoch_text),
                "lr": str(run_record["lr"]),
                "wd": str(run_record["wd"]),
                "checkpoint": checkpoint,
                "sourceExperiment": (
                    result.get("experiment")
                    or result.get("beaker")
                    or run_record.get("experiment")
                    or run_record.get("beaker")
                ),
            }
            if checkpoint is None:
                no_checkpoint.append(item)
                continue
            if checkpoint in already_registered or checkpoint in seen:
                continue
            if not re.search(r"/step\d+$", checkpoint):
                raise RuntimeError(
                    f"registered endpoint is not an exact step checkpoint: {checkpoint}"
                )
            if not item["sourceExperiment"]:
                raise RuntimeError(f"missing source experiment for {method} E{epoch_text}")
            seen.add(checkpoint)
            found.append(item)
    found.sort(
        key=lambda item: (
            item["batchSequences"],
            item["variant"],
            item["epoch"],
            float(item["lr"]),
            float(item["wd"]),
        )
    )
    return found, no_checkpoint


def task_name(item: dict[str, Any]) -> str:
    lr = item["lr"].replace(".", "p").replace("-", "m")
    wd = item["wd"].replace(".", "p")
    return f"eval-{item['variant']}-bs{item['batchSequences']}-e{item['epoch']}-lr{lr}-wd{wd}"


def group_id(item: dict[str, Any]) -> str:
    return f"bs{item['batchSequences']}-{item['variant']}"


def group_task_name(item: dict[str, Any]) -> str:
    return f"downstream-eval-{group_id(item)}"


def replace_argument(arguments: list[str], prefix: str, value: str) -> list[str]:
    matches = [index for index, argument in enumerate(arguments) if argument.startswith(prefix)]
    if len(matches) > 1:
        raise RuntimeError(f"source command contains duplicate {prefix} arguments")
    output = list(arguments)
    if matches:
        output[matches[0]] = value
    else:
        output.append(value)
    return output


def extract_source_training_command(task: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Extract the first stage command without parsing unrelated multiline shell."""
    arguments = task.get("arguments", [])
    if arguments[:2] != ["bash", "-lc"] or len(arguments) != 3:
        raise RuntimeError("source experiment is not a bash training task")
    for raw_line in arguments[2].splitlines():
        line = raw_line.strip()
        if not line.startswith("torchrun "):
            continue
        parts = shlex.split(line)
        script_indexes = [index for index, value in enumerate(parts) if value.endswith(".py")]
        if len(script_indexes) != 1 or script_indexes[0] + 1 >= len(parts):
            continue
        script_index = script_indexes[0]
        training_arguments = parts[script_index + 2 :]
        for index, value in enumerate(training_arguments):
            if value in {"|", "||", "&&", ";"} or re.fullmatch(r"(?:\d*>>?|\d*>&\d+)", value):
                training_arguments = training_arguments[:index]
                break
        return parts[script_index], parts[script_index + 1], training_arguments
    raise RuntimeError("source experiment contains no torchrun training command")


def set_env(task: dict[str, Any], name: str, value: str) -> None:
    env = task.setdefault("envVars", [])
    matches = [entry for entry in env if entry.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"source task contains duplicate {name} environment variables")
    if matches:
        matches[0].pop("secret", None)
        matches[0]["value"] = value
    else:
        env.append({"name": name, "value": value})


def disable_heldout(arguments: list[str]) -> list[str]:
    prefix = "--trainer.callbacks.heldout_evaluator="
    values = [argument for argument in arguments if argument.startswith(prefix)]
    if len(values) != 1 or "dclm_0802_validation.json" not in values[0]:
        raise RuntimeError("source command lacks the required DCLM-0802 held-out evaluator")
    heldout = values[0][len(prefix) :]
    heldout = heldout.replace("eval_on_finish: true", "eval_on_finish: false")
    heldout = heldout.replace("eval_on_startup: true", "eval_on_startup: false")
    if "eval_on_startup:" not in heldout:
        closing = heldout.rfind("}")
        if closing < 0:
            raise RuntimeError("could not edit held-out evaluator configuration")
        heldout = heldout[:closing] + ", eval_on_startup: false" + heldout[closing:]
    return replace_argument(arguments, prefix, prefix + heldout)


def evaluation_arguments(
    base_arguments: list[str], item: dict[str, Any], run_name: str
) -> list[str]:
    removable = (
        "--load_path=",
        "--load_trainer_state=",
        "--force_exact_trainer_load_path=",
        "--trainer.load_path=",
        "--trainer.load_trainer_state=",
        "--trainer.load_optim_state=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--trainer.callbacks.checkpointer.fixed_steps=",
    )
    arguments = [argument for argument in base_arguments if not argument.startswith(removable)]
    replacements = (
        ("--save-folder=", f"--save-folder=/tmp/{run_name}"),
        ("--load_path=", f"--load_path={item['checkpoint']}"),
        ("--load_trainer_state=", "--load_trainer_state=false"),
        (
            "--trainer.callbacks.wandb.name=",
            f"--trainer.callbacks.wandb.name={run_name}",
        ),
        (
            "--trainer.callbacks.wandb.tags=",
            (
                "--trainer.callbacks.wandb.tags=[checkpoint-eval,step1,0802,"
                f"downstream-nine,{item['variant']},bs{item['batchSequences']},e{item['epoch']}]"
            ),
        ),
        (
            "--trainer.callbacks.downstream_evaluator.tasks=",
            "--trainer.callbacks.downstream_evaluator.tasks=" + DOWNSTREAM_TASK_ARGUMENT,
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_interval=",
            "--trainer.callbacks.downstream_evaluator.eval_interval=null",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_startup=",
            "--trainer.callbacks.downstream_evaluator.eval_on_startup=true",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.cancel_after_first_eval=",
            "--trainer.callbacks.downstream_evaluator.cancel_after_first_eval=true",
        ),
        (
            "--trainer.callbacks.checkpointer.enabled=",
            "--trainer.callbacks.checkpointer.enabled=false",
        ),
        (
            "--train_module.validate_optimizer_hyperparameters_on_load=",
            "--train_module.validate_optimizer_hyperparameters_on_load=false",
        ),
    )
    for prefix, value in replacements:
        arguments = replace_argument(arguments, prefix, value)
    return disable_heldout(arguments)


def source_spec(experiment: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if experiment not in cache:
        cache[experiment] = json.loads(
            run(["beaker", "experiment", "spec", experiment, "--format", "json"])
        )
    return cache[experiment]


def prefetch_source_specs(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    experiments = sorted({item["sourceExperiment"] for item in items})

    def fetch(experiment: str) -> tuple[str, dict[str, Any]]:
        payload = run(["beaker", "experiment", "spec", experiment, "--format", "json"])
        return experiment, json.loads(payload)

    with ThreadPoolExecutor(max_workers=min(12, len(experiments))) as executor:
        return dict(executor.map(fetch, experiments))


def build_task(
    item: dict[str, Any], revision: str, priority: str, cache: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    base = source_spec(item["sourceExperiment"], cache)
    source_task = base["tasks"][0]
    training_script, _, base_arguments = extract_source_training_command(source_task)
    name = task_name(item)
    arguments = evaluation_arguments(base_arguments, item, name)
    task = copy.deepcopy(source_task)
    task["name"] = name
    task["arguments"] = [
        "bash",
        "-lc",
        "\n".join(
            [
                "set -euo pipefail",
                shlex.join(["test", "-d", item["checkpoint"]]),
                shlex.join(
                    [
                        "echo",
                        (
                            "DOWNSTREAM_EVAL_START "
                            f"run_id={item['runId']} epoch={item['epoch']} "
                            f"checkpoint={item['checkpoint']}"
                        ),
                    ]
                ),
                shlex.join(["python", training_script, name, "--dry-run", *arguments]),
                shlex.join(
                    [
                        "torchrun",
                        "--nproc-per-node=1",
                        training_script,
                        name,
                        *arguments,
                    ]
                ),
                shlex.join(
                    [
                        "echo",
                        f"DOWNSTREAM_EVAL_COMPLETE run_id={item['runId']} epoch={item['epoch']}",
                    ]
                ),
            ]
        ),
    ]
    task["resources"] = {"gpuCount": 1, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": False}
    for key in (
        "replicas",
        "leaderSelection",
        "propagateFailure",
        "propagatePreemption",
        "synchronizedStartTimeout",
    ):
        task.pop(key, None)
    set_env(task, "GIT_REF", revision)
    set_env(task, "NUM_NODES", "1")
    return task


def build_specs(
    items: list[dict[str, Any]], revision: str, priority: str
) -> tuple[list[tuple[str, dict[str, Any], list[dict[str, Any]]]], dict[str, Any]]:
    if not items:
        raise RuntimeError("no unevaluated retained checkpoints remain")
    if len(items) > CHECKPOINT_CAP:
        raise RuntimeError(
            f"campaign has {len(items)} checkpoints, above the audited cap of {CHECKPOINT_CAP}"
        )
    cache = prefetch_source_specs(items)
    first_base = copy.deepcopy(source_spec(items[0]["sourceExperiment"], cache))
    source_task = first_base["tasks"][0]
    training_script, _, base_arguments = extract_source_training_command(source_task)
    manifest_items = [item | {"groupId": group_id(item)} for item in items]
    manifest = {
        "trainingScript": training_script,
        "baseArguments": base_arguments,
        "items": manifest_items,
    }
    SEQUENTIAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SEQUENTIAL_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(group_id(item), []).append(item)
    specs: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for group, group_items in sorted(grouped.items()):
        base = copy.deepcopy(source_spec(group_items[0]["sourceExperiment"], cache))
        task_name_value = f"downstream-eval-{group}"
        task = copy.deepcopy(base["tasks"][0])
        task["name"] = task_name_value
        task["arguments"] = [
            "python",
            "scripts/models/run_dense_1b_downstream_sequence.py",
            "--manifest",
            str(SEQUENTIAL_MANIFEST),
            "--group-id",
            group,
        ]
        task["resources"] = {"gpuCount": 1, "sharedMemory": "10 GiB"}
        task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": False}
        for key in (
            "replicas",
            "leaderSelection",
            "propagateFailure",
            "propagatePreemption",
            "synchronizedStartTimeout",
        ):
            task.pop(key, None)
        set_env(task, "GIT_REF", revision)
        set_env(task, "NUM_NODES", "1")
        base["tasks"] = [task]
        base.pop("description", None)
        records = [
            item
            | {
                "taskName": task_name_value,
                "groupId": group,
                "status": "submitted",
            }
            for item in group_items
        ]
        specs.append((group, base, records))
    return specs, manifest


def print_plan(items: list[dict[str, Any]], no_checkpoint: list[dict[str, Any]]) -> None:
    counts = Counter((item["variant"], item["batchSequences"]) for item in items)
    print(f"independent one-GPU Beaker experiments: {len(counts)}")
    print(f"checkpoints evaluated serially: {len(items)}")
    for (name, batch), count in sorted(
        counts.items(), key=lambda entry: (entry[0][1], entry[0][0])
    ):
        print(f"  BS{batch} {name}: {count}")
    print(
        f"completed E1 records without a separately registered endpoint checkpoint: {len(no_checkpoint)}"
    )
    print(
        "  These are inherited Original E1 coordinates; their existing baseline downstream results are reused by the report."
    )


def register_campaign(
    report: dict[str, Any], name: str, experiment: str, revision: str, tasks: list[dict[str, Any]]
) -> None:
    campaigns = report.setdefault("downstreamEvaluationCampaigns", [])
    if any(campaign.get("experiment") == experiment for campaign in campaigns):
        raise RuntimeError(f"campaign {experiment} is already registered")
    campaigns.append(
        {
            "id": name,
            "experiment": experiment,
            "revision": revision,
            "status": "submitted",
            "gpuPerTask": 1,
            "taskCount": len({task["taskName"] for task in tasks}),
            "checkpointCount": len(tasks),
            "singleGpuSequential": True,
            "independentExperiment": True,
            "runtimeDirectoryDiscovery": True,
            "tasks": tasks,
        }
    )


def normalize_task(name: str) -> str:
    for prefix in ("csqa", "openbookqa", "socialiqa"):
        if name.startswith(prefix):
            return prefix
    return name


def parse_metrics(logs: str) -> dict[str, Any]:
    clean = ANSI.sub("", logs)
    accuracy: dict[str, float] = {}
    bpb: dict[str, float] = {}
    for task, value in ACCURACY.findall(clean):
        accuracy[normalize_task(task)] = 100 * float(value)
    for task, value in BPB.findall(clean):
        bpb[normalize_task(task)] = float(value)
    missing = [task for task in REPORT_TASKS if task not in accuracy or task not in bpb]
    if missing:
        raise RuntimeError(f"completed downstream task is missing metrics for {missing}")
    result: dict[str, Any] = {
        "acc": accuracy["hellaswag"],
        "bpb": bpb["hellaswag"],
        "avg8Bpb": sum(bpb[task] for task in AVERAGE_TASKS) / len(AVERAGE_TASKS),
        "avg8Accuracy": sum(accuracy[task] for task in AVERAGE_TASKS) / len(AVERAGE_TASKS),
        "downstream": accuracy,
        "downstreamBpb": bpb,
    }
    wandb = WANDB_RUN.findall(clean)
    if wandb:
        result["downstreamWandb"] = wandb[-1]
    return result


def job_state(job: dict[str, Any]) -> str:
    status = job.get("status") or {}
    if "finalized" in status:
        return "complete" if status.get("exitCode") == 0 else "failed"
    if "started" in status:
        return "running"
    if "scheduled" in status:
        return "scheduled"
    return "submitted"


def find_result(report: dict[str, Any], run_id: str, epoch: int) -> dict[str, Any]:
    runs = [run_record for run_record in report.get("runs", []) if run_record.get("id") == run_id]
    if len(runs) != 1:
        raise RuntimeError(f"expected one report run {run_id}; found {len(runs)}")
    result = (runs[0].get("results") or {}).get(str(epoch))
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise RuntimeError(f"registered endpoint {run_id} E{epoch} is no longer complete")
    return result


def inspect_experiment(experiment: str) -> dict[str, Any]:
    payload = json.loads(run(["beaker", "experiment", "inspect", experiment, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one experiment for {experiment}")
    return payload[0]


def matching_job(jobs: Iterable[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [
        job
        for job in jobs
        if job.get("name") == name or str(job.get("name", "")).startswith(name + "-replica-")
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple Beaker jobs match task {name}")
    return matches[0] if matches else None


def refresh(report: dict[str, Any]) -> bool:
    changed = False
    for campaign in report.get("downstreamEvaluationCampaigns", []):
        if campaign.get("status") not in ACTIVE:
            continue
        inspected = inspect_experiment(campaign["experiment"])
        jobs = inspected.get("jobs") or []
        sequential = bool(
            campaign.get("singleGpuSequential") or campaign.get("groupedSingleGpuSequential")
        )
        log_cache: dict[str, str] = {}
        for task in campaign.get("tasks", []):
            if task.get("status") == "complete":
                continue
            job = matching_job(jobs, task["taskName"])
            if job is None:
                continue
            state = job_state(job)
            sequential_logs = ""
            if sequential and state in {"running", "complete", "failed"}:
                if job["id"] not in log_cache:
                    log_cache[job["id"]] = run(
                        ["beaker", "job", "logs", job["id"], "--no-timestamps"]
                    )
                sequential_logs = log_cache[job["id"]]
            if not sequential and task.get("status") != state:
                task["status"] = state
                changed = True
            task["job"] = job["id"]
            marker = f"run_id={task['runId']} epoch={task['epoch']}"
            complete_marker = f"DOWNSTREAM_EVAL_COMPLETE {marker}"
            skipped_marker = f"DOWNSTREAM_EVAL_SKIPPED {marker}"
            if sequential:
                if skipped_marker in sequential_logs:
                    if task.get("status") != "unavailable":
                        task["status"] = "unavailable"
                        task["reason"] = "Registered checkpoint directory is missing."
                        changed = True
                    continue
                if complete_marker not in sequential_logs:
                    checkpoint_state = "submitted"
                    if f"DOWNSTREAM_EVAL_START {marker}" in sequential_logs:
                        checkpoint_state = "running" if state != "failed" else "failed"
                    if task.get("status") != checkpoint_state:
                        task["status"] = checkpoint_state
                        changed = True
                    continue
                start = sequential_logs.rfind(
                    f"DOWNSTREAM_EVAL_START {marker}", 0, sequential_logs.find(complete_marker)
                )
                logs = sequential_logs[
                    start : sequential_logs.find(complete_marker) + len(complete_marker)
                ]
                resolved = re.search(rf"DOWNSTREAM_EVAL_START {re.escape(marker)} checkpoint=(\S+)", logs)
                if resolved and task.get("checkpoint") != resolved.group(1):
                    task["registeredCheckpoint"] = task.get("checkpoint")
                    task["checkpoint"] = resolved.group(1)
                    changed = True
                if task.get("status") != "complete":
                    task["status"] = "complete"
                    changed = True
            else:
                if state != "complete":
                    continue
                logs = run(["beaker", "job", "logs", job["id"], "--no-timestamps"])
            metrics = parse_metrics(logs)
            result = find_result(report, task["runId"], int(task["epoch"]))
            result.update(metrics)
            result["downstreamEvaluation"] = {
                "campaign": campaign["experiment"],
                "job": job["id"],
                "checkpoint": task["checkpoint"],
                "gpuCount": 1,
            }
            changed = True
        states = [task.get("status") for task in campaign.get("tasks", [])]
        new_status = (
            "complete"
            if states and all(state == "complete" for state in states)
            else "complete_with_missing"
            if states
            and any(state == "unavailable" for state in states)
            and all(state in {"complete", "unavailable"} for state in states)
            else "failed"
            if any(state == "failed" for state in states)
            else "running"
            if any(state == "running" for state in states)
            else "scheduled"
            if any(state == "scheduled" for state in states)
            else "submitted"
        )
        if campaign.get("status") != new_status:
            campaign["status"] = new_status
            changed = True
    return changed


def main() -> None:
    args = parse_args()
    report = load_report(args.report)
    if args.refresh:
        changed = refresh(report)
        if changed:
            write_report(report, args.report)
        campaigns = report.get("downstreamEvaluationCampaigns", [])
        print(
            "\n".join(
                f"{campaign.get('experiment')}: {campaign.get('status')} "
                f"({sum(task.get('status') == 'complete' for task in campaign.get('tasks', []))}/"
                f"{len(campaign.get('tasks', []))} complete)"
                for campaign in campaigns
            )
            or "no registered downstream-evaluation campaign"
        )
        return

    if args.replace_deleted_experiment:
        campaigns = report.get("downstreamEvaluationCampaigns", [])
        matches = [c for c in campaigns if c.get("experiment") == args.replace_deleted_experiment]
        if len(matches) != 1:
            raise RuntimeError("deleted experiment registration was not found exactly once")
        report["downstreamEvaluationCampaigns"] = [
            c for c in campaigns if c.get("experiment") != args.replace_deleted_experiment
        ]
    ignored = {args.supersede_experiment} if args.supersede_experiment else set()
    if args.supersede_experiment:
        matches = [
            campaign
            for campaign in report.get("downstreamEvaluationCampaigns", [])
            if campaign.get("experiment") == args.supersede_experiment
        ]
        if len(matches) != 1 or matches[0].get("status") != "failed":
            raise RuntimeError("--supersede-experiment must identify exactly one failed campaign")
    items, no_checkpoint = candidates(report, ignored_experiments=ignored)
    if args.plan:
        print_plan(items, no_checkpoint)
        return
    specs, manifest = build_specs(items, args.revision, args.priority)
    if args.print_spec:
        json.dump(
            [{"groupId": group, "spec": spec} for group, spec, _ in specs],
            sys.stdout,
            indent=2,
        )
        print()
        return
    submitted: list[str] = []
    for group, spec, task_records in specs:
        experiment_name = f"{args.name}-{group}"
        output = run(
            [
                "beaker",
                "experiment",
                "create",
                "-",
                "--name",
                experiment_name,
                "--workspace",
                args.workspace,
                "--priority",
                args.priority,
            ],
            input_text=json.dumps(spec),
        )
        print(output, end="")
        ids = ULID.findall(output)
        if len(ids) != 1:
            raise RuntimeError("submission succeeded but did not return exactly one experiment ID")
        register_campaign(
            report, experiment_name, ids[0], args.revision, task_records
        )
        submitted.append(ids[0])
        write_report(report, args.report)
    if args.supersede_experiment:
        old = next(
            campaign
            for campaign in report["downstreamEvaluationCampaigns"]
            if campaign.get("experiment") == args.supersede_experiment
        )
        old["status"] = "superseded"
        old["supersededBy"] = submitted
        write_report(report, args.report)


if __name__ == "__main__":
    main()
