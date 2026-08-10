#!/usr/bin/env python3
"""Submit one guarded nested-3B E1 continuation from a 1B pre-decay checkpoint."""

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


SEQ = 4096
POOL_TOKENS = 3_000_000_000
DECAY = 0.1
ROOT = "/weka/oe-training-default/sewonm/icsl/data/dclm_0802_nested_1b_3b_9b"
EXTENSION = f"{ROOT}/manifests/dclm_0802_nested_extension_1b_to_3b.json"
POOL_MANIFEST = f"{ROOT}/manifests/dclm_0802_nested_1b_3b_9b.pool.json"
MODELS = {
    "153m": {"lr": "2e-3", "rank_mb": 16, "report": "wsd_batch_size_153m_pool3b.json"},
    "474m": {"lr": "2e-3", "rank_mb": 16, "report": "wsd_batch_size_474m_pool3b.json"},
    "1b": {"lr": "1e-3", "rank_mb": 4, "report": "wsd_batch_size_1b_pool3b.json"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--global-sequences", choices=(64, 128, 256), type=int, required=True)
    p.add_argument("--weight-decay", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--base-experiment", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--suffix", required=True)
    p.add_argument("--workspace", default="ai2/flex2")
    p.add_argument("--priority", default="urgent")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--register", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", a.suffix):
        p.error("suffix must be a lowercase run-name component")
    expected = stable_step(1_000_000_000, a.global_sequences)
    if not a.source_checkpoint.endswith(f"/step{expected}"):
        p.error(f"source must be the exact 1B pre-decay checkpoint /step{expected}")
    return a


def nproc(model: str, batch: int) -> int:
    if model in {"153m", "474m"} and batch == 64:
        return 4
    return 8


def total_steps(tokens: int, batch: int) -> int:
    return math.ceil(tokens / (batch * SEQ))


def stable_step(tokens: int, batch: int) -> int:
    end = total_steps(tokens, batch)
    return end - round(DECAY * end) - 1


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


def audit(a: argparse.Namespace, report: dict[str, Any]) -> None:
    plan = report["poolPlan"]
    batch = str(a.global_sequences)
    lr = MODELS[a.model]["lr"]
    if Decimal(plan["fixedLearningRateByBatch"][batch]) != Decimal(lr):
        raise SystemExit("report LR policy does not match launcher")
    allowed = {Decimal(x) for x in plan["initialWeightDecayByBatch"][batch]}
    if Decimal(a.weight_decay) not in allowed:
        raise SystemExit(f"WD {a.weight_decay} is outside initial candidates {sorted(allowed)}")
    for sweep in report.get("batchSweeps", []):
        if (
            sweep.get("batchSequences") == a.global_sequences
            and Decimal(str(sweep.get("lr"))) == Decimal(lr)
            and Decimal(str(sweep.get("wd"))) == Decimal(a.weight_decay)
            and sweep.get("activeEpoch") == 1
        ):
            raise SystemExit("refusing duplicate registered E1 tuple")


def build(a: argparse.Namespace, source: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    spec = copy.deepcopy(source)
    task = spec["tasks"][0]
    original = task["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("expected a single-endpoint Gantry Python task")
    script = original[1]
    args = original[3:]
    blocked = (
        "--load_path=", "--load_trainer_state=", "--trainer.load_path=",
        "--trainer.load_trainer_state=", "--trainer.load_optim_state=",
        "--trainer.reset_data_loader_state_on_load_path=",
        "--trainer.callbacks.checkpointer.save_interval=",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
    )
    args = [item for item in args if not item.startswith(blocked)]
    lr = MODELS[a.model]["lr"]
    warmup = 24 * 1024 // a.global_sequences
    name = (
        f"dense_{a.model}_step1_pool3b_nested_bs{a.global_sequences}_e1_"
        f"lr{lr}_wd{a.weight_decay}_warmup{warmup}_{a.suffix}"
    )
    output = f"/weka/oe-training-default/sewonm/icsl/models/{name}"
    replacements = (
        ("--save-folder=", f"--save-folder={output}"),
        ("--dataset.subset_manifest=", f"--dataset.subset_manifest={EXTENSION}"),
        ("--dataset.mix=", "--dataset.mix=null"),
        ("--trainer.max_duration=", f"--trainer.max_duration={{value: {POOL_TOKENS}, unit: tokens}}"),
        ("--trainer.callbacks.wandb.name=", f"--trainer.callbacks.wandb.name={name}"),
        ("--trainer.callbacks.wandb.tags=", "--trainer.callbacks.wandb.tags=" +
         f"[pretraining,step1,nested-3b,new-2b-loader-reset,dense-{a.model},bs{a.global_sequences},wsd]"),
        ("--trainer.callbacks.checkpointer.fixed_steps=", f"--trainer.callbacks.checkpointer.fixed_steps=[{stable_step(POOL_TOKENS, a.global_sequences)}]"),
        ("--data_loader.global_batch_size=", f"--data_loader.global_batch_size={a.global_sequences * SEQ}"),
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
        shlex.join(["test", "-f", EXTENSION]),
        shlex.join(["test", "-f", POOL_MANIFEST]),
        "python - <<'PY'\nimport json\np=json.load(open('" + POOL_MANIFEST + "'))\n"
        "assert p['audit']['passed'] is True\nassert p['audit']['base_document_overlap']==0\n"
        "assert p['audit']['chunk_document_overlap']==0\nprint('POOL3B_AUDIT_OK')\nPY",
        shlex.join(["test", "!", "-e", output]),
        shlex.join(["echo", f"POOL3B_E1_PREFLIGHT model={a.model} bs={a.global_sequences} lr={lr} wd={a.weight_decay} source={a.source_checkpoint} reset_loader=true"]),
        shlex.join(["python", script, name, "--dry-run", *args]),
        shlex.join(["torchrun", f"--nproc-per-node={nproc(a.model, a.global_sequences)}", script, name, *args]),
    ]
    task["arguments"] = ["bash", "-lc", "\n".join(checks)]
    task["envVars"] = [x for x in task.get("envVars", []) if x.get("name") != "GANTRY_USE_TORCHRUN"]
    for env in task["envVars"]:
        if env.get("name") == "GIT_REF":
            env["value"] = a.revision
            break
    task["resources"] = {"gpuCount": nproc(a.model, a.global_sequences), "sharedMemory": "10 GiB"}
    task["context"] = {"priority": a.priority, "minRuntime": "0s", "autoResume": False}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec, name, output


def register(a: argparse.Namespace, experiment: str, output: str) -> None:
    path = report_path(a.model)
    report = json.loads(path.read_text())
    gpus = nproc(a.model, a.global_sequences)
    rank_mb = MODELS[a.model]["rank_mb"]
    report["batchSweeps"].append({
        "batchSequences": a.global_sequences,
        "globalBatchTokens": a.global_sequences * SEQ,
        "contextLength": SEQ,
        "lr": MODELS[a.model]["lr"],
        "wd": a.weight_decay,
        "warmupSteps": 24 * 1024 // a.global_sequences,
        "rankMicrobatchSequences": rank_mb,
        "gradientAccumulation": a.global_sequences // (gpus * rank_mb),
        "gpuCount": gpus,
        "status": "pending",
        "activeEpoch": 1,
        "search": "nested-3b-fixed-lr-adaptive-wd",
        "beaker": experiment,
        "output": output,
        "sourceCheckpoint": a.source_checkpoint,
        "dataManifest": EXTENSION,
        "dataLoaderReset": True,
        "retainedPreDecaySteps": [stable_step(POOL_TOKENS, a.global_sequences)],
        "actualTargetTokens": POOL_TOKENS,
        "results": {},
    })
    report["updated"] = "2026-08-09"
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".js").write_text("window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n")


def main() -> None:
    a = parse_args()
    report = json.loads(report_path(a.model).read_text())
    audit(a, report)
    spec, name, output = build(a, base_spec(a.base_experiment))
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
