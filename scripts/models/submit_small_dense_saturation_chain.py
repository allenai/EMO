#!/usr/bin/env python3
"""Submit one persistent BS32 saturation chain for Dense 153M or 474M."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = "ai2/flex2"
SEQUENCE_LENGTH = 4096
GLOBAL_SEQUENCES = 32
RANK_MICROBATCH_SEQUENCES = 16
WARMUP_STEPS = 768
DOWNSTREAM_TASKS = (
    "[arc_easy, arc_challenge, boolq, csqa_val_rc_5shot, hellaswag, "
    "openbookqa_test_rc_5shot, piqa, socialiqa_val_rc_5shot, winogrande]"
)
MANIFEST_DIR = Path("scripts/models/manifests")
MODELS: dict[str, dict[str, Any]] = {
    "474m": {
        "baseExperiment": "01KZ7307CK7ZZQ1XCJ2QQ08KD4",
        "report": Path("reports/0802/data/wsd_batch_size_474m.json"),
        "previousEpoch": 24,
        "firstEpoch": 32,
        "previousValidation": 3.119,
        "learningRate": "1e-3",
        "weightDecay": "0.1",
        "output": (
            "/weka/oe-training-default/sewonm/icsl/models/"
            "dense_474m_dclm1b/bs32_lr1e-3_wd0.1"
        ),
        "sourceStep": 164794,
        "runSuffix": "sm0818bs32saturation",
        "name": "dense-474m-bs32-saturation-chain",
    },
    "153m": {
        "baseExperiment": "01KZ6Q4DJ8J994A6SQ39MEGTZ2",
        "report": Path("reports/0802/data/wsd_batch_size_153m.json"),
        "previousEpoch": 48,
        "firstEpoch": 56,
        "previousValidation": 3.371,
        "learningRate": "1e-3",
        "weightDecay": "0.033",
        "output": (
            "/weka/oe-training-default/sewonm/icsl/models/"
            "dense_153m_dclm1b/bs32_lr1e-3_wd0.033"
        ),
        "sourceStep": 329589,
        "runSuffix": "sm0818bs32saturation",
        "name": "dense-153m-bs32-saturation-chain",
    },
}
REMOVE_PREFIXES = (
    "--save-folder=",
    "--trainer.max_duration=",
    "--trainer.callbacks.wandb.name=",
    "--trainer.callbacks.wandb.tags=",
    "--trainer.callbacks.checkpointer.fixed_steps=",
    "--data_loader.global_batch_size=",
    "--train_module.rank_microbatch_size=",
    "--train_module.scheduler=",
    "--train_module.optim.weight_decay=",
    "--lr=",
    "--load_path=",
    "--load_trainer_state=",
    "--force_exact_trainer_load_path=",
    "--trainer.load_path=",
    "--trainer.load_trainer_state=",
    "--trainer.load_optim_state=",
    "--trainer.reset_data_loader_state_on_load_path=",
    "--train_module.validate_optimizer_hyperparameters_on_load=",
    "--model.tie_embeddings=",
    "--trainer.callbacks.checkpointer.save_interval=",
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
)
FORBIDDEN_EXACT = {
    "--dynamic-repacking",
    "--batch-shuffling",
    "--fixed-data-order",
    "--no-data-shuffle",
    "--decay-embeddings",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--revision", required=True, help="Pushed, fetchable git revision")
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Replace the registered failed logical chain while retaining attempt provenance",
    )
    return parser.parse_args()


def run(arguments: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        arguments,
        check=True,
        input=input_text,
        capture_output=True,
        text=True,
    )
    return result.stdout


def base_spec(experiment: str) -> dict[str, Any]:
    return json.loads(run(["beaker", "experiment", "spec", experiment, "--format", "json"]))


def upsert(arguments: list[str], prefix: str, replacement: str) -> list[str]:
    found = False
    output: list[str] = []
    for argument in arguments:
        if argument.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
        else:
            output.append(argument)
    if not found:
        output.append(replacement)
    return output


def manifest_path(model: str) -> Path:
    return MANIFEST_DIR / f"small-dense-bs32-{model}-saturation.json"


def build_manifest(model: str, spec: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    settings = MODELS[model]
    original = spec["tasks"][0]["arguments"]
    if original[:1] != ["python"]:
        raise ValueError("expected a Gantry Python source task")
    script = original[1]
    arguments = [
        argument
        for argument in original[3:]
        if argument not in FORBIDDEN_EXACT
        and not argument.startswith(REMOVE_PREFIXES)
        and not argument.startswith("--mlp-weight-decay")
    ]
    replacements = (
        ("--save-folder=", f"--save-folder={settings['output']}"),
        (
            "--data_loader.global_batch_size=",
            f"--data_loader.global_batch_size={GLOBAL_SEQUENCES * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.rank_microbatch_size=",
            f"--train_module.rank_microbatch_size={RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH}",
        ),
        (
            "--train_module.optim.weight_decay=",
            f"--train_module.optim.weight_decay={settings['weightDecay']}",
        ),
        ("--lr=", f"--lr={settings['learningRate']}"),
        ("--model.tie_embeddings=", "--model.tie_embeddings=false"),
        (
            "--trainer.callbacks.downstream_evaluator.tasks=",
            f"--trainer.callbacks.downstream_evaluator.tasks={DOWNSTREAM_TASKS}",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_interval=",
            "--trainer.callbacks.downstream_evaluator.eval_interval=null",
        ),
        (
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=true",
        ),
        (
            "--trainer.callbacks.checkpointer.save_interval=",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
        ),
        (
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        ),
    )
    for prefix, replacement in replacements:
        arguments = upsert(arguments, prefix, replacement)

    source = f"{settings['output']}/step{settings['sourceStep']}"
    manifest = {
        "model": model,
        "script": script,
        "output": settings["output"],
        "baseArguments": arguments,
        "globalSequences": GLOBAL_SEQUENCES,
        "nprocPerNode": 2,
        "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
        "warmupSteps": WARMUP_STEPS,
        "learningRate": settings["learningRate"],
        "weightDecay": settings["weightDecay"],
        "previousEpoch": settings["previousEpoch"],
        "firstEpoch": settings["firstEpoch"],
        "epochIncrement": 8,
        "previousValidation": settings["previousValidation"],
        "initialSourceCheckpoint": source,
        "runSuffix": settings["runSuffix"],
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
    }
    path = manifest_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, path


def set_revision(task: dict[str, Any], revision: str) -> None:
    for environment_variable in task.get("envVars", []):
        if environment_variable.get("name") == "GIT_REF":
            environment_variable["value"] = revision
            return
    raise ValueError("source task has no GIT_REF")


def build_submission(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    settings = MODELS[args.model]
    source_spec = base_spec(settings["baseExperiment"])
    manifest, path = build_manifest(args.model, source_spec)
    spec = copy.deepcopy(source_spec)
    task = spec["tasks"][0]
    task["arguments"] = [
        "python",
        "scripts/models/run_small_dense_saturation_chain.py",
        "--manifest",
        str(path),
    ]
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in {"GANTRY_USE_TORCHRUN", "PYTORCH_CUDA_ALLOC_CONF"}
    ]
    set_revision(task, args.revision)
    if args.model == "474m":
        task["envVars"].append(
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"}
        )
    task["resources"] = {"gpuCount": 2, "sharedMemory": "10 GiB"}
    task["context"] = {
        "priority": args.priority,
        "minRuntime": "0s",
        "autoResume": True,
    }
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec["retry"] = {"allowedTaskRetries": 8}
    spec["description"] = (
        f"Persistent Dense {args.model.upper()} BS32 WSD saturation chain from "
        f"E{settings['previousEpoch']}; ordinary data order, untied weights, and zero "
        "embedding WD; one canonical output directory; strict validation non-improvement stop."
    )
    return spec, manifest, path


def audit_name(workspace: str, name: str) -> None:
    payload = json.loads(
        run(["beaker", "workspace", "experiments", workspace, "--text", name, "--format", "json"])
    )
    experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
    if any(experiment.get("name") == name for experiment in experiments):
        raise SystemExit(f"refusing duplicate Beaker experiment name {name}")


def write_report(
    model: str,
    experiment: str,
    revision: str,
    manifest: dict[str, Any],
    *,
    resume_failed: bool = False,
) -> None:
    settings = MODELS[model]
    report_path = settings["report"]
    report = json.loads(report_path.read_text())
    chains = [
        sweep
        for sweep in report.get("batchSweeps", [])
        if sweep.get("saturationChain") is True
    ]
    active = [
        sweep
        for sweep in chains
        if str(sweep.get("status", "")).lower() in {"submitted", "scheduled", "running"}
    ]
    if active:
        raise RuntimeError(f"an active saturation chain is already registered: {active}")
    source = manifest["initialSourceCheckpoint"]
    sweep = {
        "id": f"dense-{model}-bs32-persistent-saturation",
        "batchSequences": GLOBAL_SEQUENCES,
        "globalBatchTokens": GLOBAL_SEQUENCES * SEQUENCE_LENGTH,
        "contextLength": SEQUENCE_LENGTH,
        "lr": settings["learningRate"],
        "wd": settings["weightDecay"],
        "warmupSteps": WARMUP_STEPS,
        "rankMicrobatchSequences": RANK_MICROBATCH_SEQUENCES,
        "gradientAccumulation": 1,
        "gpuCount": 2,
        "status": "submitted",
        "activeEpoch": settings["firstEpoch"],
        "search": "small-model-persistent-saturation",
        "saturationChain": True,
        "stopOnNonImprovement": True,
        "epochIncrement": 8,
        "beaker": experiment,
        "experiment": experiment,
        "revision": revision,
        "output": settings["output"],
        "canonicalOutput": settings["output"],
        "sourceCheckpoint": source,
        "frontierEpoch": settings["previousEpoch"],
        "frontierValidation": settings["previousValidation"],
        "dynamicRepacking": False,
        "weightTying": False,
        "embeddingWeightDecay": "zero",
        "automaticTaskRetries": 8,
        "results": {},
        "reason": (
            f"Submitted one persistent job from the healthy E{settings['previousEpoch']} "
            "pre-decay checkpoint. Successive eight-epoch WSD endpoints share the same "
            "canonical output directory and stop at the first held-out validation "
            "non-improvement. Beaker task retries and atomic stage markers recover "
            "infrastructure failures automatically."
        ),
    }
    if resume_failed:
        if not chains:
            raise RuntimeError("no failed saturation chain is registered for recovery")
        previous = chains[-1]
        if str(previous.get("status", "")).lower() != "failed":
            raise RuntimeError(f"registered saturation chain is not failed: {previous}")
        previous_experiment = str(previous["experiment"])
        history = list(previous.get("attemptHistory", []))
        history.append(
            {
                "beaker": previous_experiment,
                "status": "failed",
                "failureClass": "preflight-output-directory-semantic",
                "reason": (
                    "The launcher used the legacy run-name output directory instead of the "
                    "canonical model/coordinate directory and exhausted automatic retries "
                    "before training started."
                ),
            }
        )
        previous.clear()
        previous.update(sweep)
        previous["attemptHistory"] = history
        previous["recoveryOf"] = previous_experiment
        previous["reason"] = (
            f"Recovered the failed launcher from the verified E{settings['previousEpoch']} "
            "pre-decay checkpoint under the current canonical output-directory semantic. "
            "Successive eight-epoch endpoints remain in this one directory."
        )
    else:
        report.setdefault("batchSweeps", []).append(sweep)
    note = (
        f" BS32 now has one persistent saturation-gated continuation from "
        f"E{settings['previousEpoch']}; it uses the locked LR{settings['learningRate']}/"
        f"WD{settings['weightDecay']} coordinate and one canonical output directory."
    )
    if note.strip() not in str(report.get("selection", "")):
        report["selection"] = str(report.get("selection", "")).rstrip() + note
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    serialized = json.dumps(report, indent=2) + "\n"
    report_path.write_text(serialized)
    report_path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    args = parse_args()
    if args.resume_failed and not args.register:
        raise SystemExit("--resume-failed requires --register")
    settings = MODELS[args.model]
    name = args.name or settings["name"]
    if args.register and not args.resume_failed:
        report = json.loads(settings["report"].read_text())
        if any(
            sweep.get("saturationChain") is True
            and str(sweep.get("status", "")).lower() in {"submitted", "scheduled", "running"}
            for sweep in report.get("batchSweeps", [])
        ):
            raise SystemExit("refusing a duplicate active saturation chain")
    spec, manifest, _ = build_submission(args)
    audit_name(args.workspace, name)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    output = run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            name,
            "--workspace",
            args.workspace,
        ],
        input_text=json.dumps(spec),
    )
    print(output, end="")
    ids = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if args.register:
        if not ids:
            raise RuntimeError("submission succeeded without a parsed experiment ID")
        write_report(
            args.model,
            ids[0],
            args.revision,
            manifest,
            resume_failed=args.resume_failed,
        )


if __name__ == "__main__":
    main()
