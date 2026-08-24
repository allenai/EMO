#!/usr/bin/env python3
"""Continue one locked-WD small-model chain using POST evaluations only.

The chain resumes constant-LR training from an exact user-approved boundary,
saves every subsequent scheduled frontier, independently WSD-decays/evaluates
that frontier, and stops at the first strict POST non-improvement.  Constant-LR
frontiers are checkpoint provenance only and are never evaluated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_small_dense_dr_wt_embedwd_chain as adaptive
import run_small_dense_locked_wd_predecay_chain as locked
import run_small_dense_saturation_chain as common

POLICY = "locked_wd_all_postdecay_saturation_v1"
POST_DECAY_SOURCE_COUNT = 3


def state_dir(config: dict[str, Any]) -> Path:
    return Path(str(config["continuationRoot"]))


def constant_output(config: dict[str, Any]) -> Path:
    return state_dir(config) / "constant_lr"


def post_result_path(config: dict[str, Any], epoch: int) -> Path:
    return state_dir(config) / "post_decay" / f"e{epoch}.result.json"


def selection_path(config: dict[str, Any]) -> Path:
    return state_dir(config) / "post_decay_selection.json"


def next_epoch(config: dict[str, Any], epoch: int) -> int:
    candidate = epoch + int(config["epochIncrement"])
    adaptive.targets_through(config, candidate)
    return candidate


def prior_results(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(epoch): dict(result)
        for epoch, result in config["priorPostDecayResults"].items()
    }


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model",
        "globalSequences",
        "nprocPerNode",
        "rankMicrobatchSequences",
        "warmupSteps",
        "learningRate",
        "wdLadder",
        "initialTargets",
        "epochIncrement",
        "outputRoot",
        "runSuffix",
        "lockedWd",
        "policy",
        "comparisonPolicy",
        "preDecayEvaluation",
        "postDecaySourceCount",
        "postDecaySaturationCriterion",
        "resumeEpoch",
        "resumeCheckpoint",
        "continuationRoot",
        "priorPostDecayResults",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"manifest is missing {missing}")
    if config["policy"] != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if config["comparisonPolicy"] != "post_decay_only":
        raise ValueError("continuation must compare POST only")
    if config["preDecayEvaluation"] is not False:
        raise ValueError("pre-decay evaluation must remain disabled")
    if int(config["postDecaySourceCount"]) != POST_DECAY_SOURCE_COUNT:
        raise ValueError("postDecaySourceCount must remain exactly three")
    if config["postDecaySaturationCriterion"] != "strict_non_improvement":
        raise ValueError("POST saturation must use strict non-improvement")
    locked_wd = str(config["lockedWd"])
    if locked_wd not in [str(value) for value in config["wdLadder"]]:
        raise ValueError("lockedWd must remain on the historical WD ladder")
    if Decimal(locked_wd) > adaptive.MAX_WEIGHT_DECAY:
        raise ValueError("lockedWd must not exceed 1.0")
    resume_epoch = int(config["resumeEpoch"])
    adaptive.targets_through(config, resume_epoch)
    expected_step = adaptive.stable_step(resume_epoch, int(config["globalSequences"]))
    if Path(str(config["resumeCheckpoint"])).name != f"step{expected_step}":
        raise ValueError("resumeCheckpoint is not the exact scheduled boundary step")
    continuation_root = Path(str(config["continuationRoot"]))
    if not continuation_root.is_absolute() or ".." in continuation_root.parts:
        raise ValueError("continuationRoot must be a normalized absolute path")
    model_root = Path(str(config["outputRoot"]))
    if continuation_root.parent != model_root:
        raise ValueError("continuationRoot must be an isolated child of outputRoot")
    results = prior_results(config)
    if len(results) < POST_DECAY_SOURCE_COUNT or max(results) != resume_epoch:
        raise ValueError("prior POST provenance must contain the resume boundary")
    for epoch, result in results.items():
        if epoch > resume_epoch or int(result.get("epoch", epoch)) != epoch:
            raise ValueError("prior POST epoch provenance is inconsistent")
        if result.get("comparisonGroup") != "post_decay":
            raise ValueError("prior results must belong to the POST comparison group")
        if str(result.get("wd")) != locked_wd:
            raise ValueError("prior POST result WD does not match lockedWd")
        Decimal(str(result["validationExact"]))
        if not str(result.get("checkpoint", "")).startswith(
            str(model_root).rstrip("/") + "/"
        ):
            raise ValueError("prior POST checkpoint is outside outputRoot")


def checkpoint_for(config: dict[str, Any], epoch: int) -> Path:
    if epoch == int(config["resumeEpoch"]):
        return Path(str(config["resumeCheckpoint"]))
    return constant_output(config) / (
        f"step{adaptive.stable_step(epoch, int(config['globalSequences']))}"
    )


def run_name(config: dict[str, Any], phase: str, epoch: int) -> str:
    return (
        f"dense_{config['model']}_step1_0802_repeated_dclm1b_"
        f"bs{config['globalSequences']}_dr_wt_embwd_{phase}_e{epoch}_"
        f"lr{config['learningRate']}_wd{config['lockedWd']}_{config['runSuffix']}"
    )


def constant_training_arguments(
    config: dict[str, Any], epoch: int, source: Path, name: str
) -> list[str]:
    batch = int(config["globalSequences"])
    target_step = adaptive.stable_step(epoch, batch)
    arguments = locked.base_arguments(config)
    arguments = locked.replace_argument(
        arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
    )
    heldout = adaptive.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false"
    )
    arguments = locked.replace_argument(
        arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    arguments.extend(
        (
            f"--save-folder={constant_output(config)}",
            f"--trainer.max_duration={{value: {target_step}, unit: steps}}",
            f"--trainer.callbacks.wandb.name={name}",
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,constant-lr,checkpoint-only,locked-wd,post-only-policy]",
            f"--trainer.callbacks.checkpointer.fixed_steps=[{target_step}]",
            "--data_loader.restore_data_order_from_state=false",
            "--data_loader.ignore_fingerprint_mismatch=true",
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.ConstantScheduler}",
            "--dynamic-repacking",
            "--trainer.callbacks.checkpointer.save_interval=1000000000",
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
            "--force_exact_trainer_load_path=true",
            f"--trainer.load_path={source}",
            "--trainer.load_trainer_state=true",
            "--trainer.load_optim_state=true",
            "--trainer.reset_data_loader_state_on_load_path=false",
            "--train_module.validate_optimizer_hyperparameters_on_load=true",
        )
    )
    return arguments


def train_frontier(config: dict[str, Any], previous_epoch: int, epoch: int) -> Path:
    target = checkpoint_for(config, epoch)
    if target.is_dir():
        return target
    source = checkpoint_for(config, previous_epoch)
    if not source.is_dir():
        raise FileNotFoundError(f"missing exact constant-LR source checkpoint {source}")
    name = run_name(config, "constant_checkpoint_only", epoch)
    arguments = constant_training_arguments(config, epoch, source, name)
    print(
        f"SMALL_PREDECAY_POLICY_STAGE_START model={config['model']} "
        f"bs={config['globalSequences']} phase=constant_train epoch={epoch} "
        f"wd={config['lockedWd']} source={source} output={constant_output(config)}",
        flush=True,
    )
    subprocess.run(
        ["python", adaptive.TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True
    )
    log_path = state_dir(config) / "logs" / f"constant_train_e{epoch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={config['nprocPerNode']}",
                adaptive.TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )
    if not target.is_dir():
        raise FileNotFoundError(f"constant-LR stage did not create {target}")
    return target


def evaluation_arguments(
    config: dict[str, Any], checkpoint: Path, output: Path, name: str
) -> list[str]:
    arguments = locked.evaluation_arguments(config, checkpoint, output, name)
    return locked.replace_argument(
        arguments,
        "--trainer.callbacks.wandb.tags=",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,checkpoint-eval,post-decay,locked-wd,post-only-policy]",
    )


def emit_result(config: dict[str, Any], result: dict[str, Any]) -> None:
    locked.emit_result(config, result)


def evaluate_endpoint(
    config: dict[str, Any], epoch: int, endpoint: Path
) -> dict[str, Any]:
    output = state_dir(config) / "eval_runs" / "post_decay" / f"e{epoch}"
    name = run_name(config, "postdecay_eval", epoch)
    arguments = evaluation_arguments(config, endpoint, output, name)
    print(
        f"SMALL_PREDECAY_POLICY_STAGE_START model={config['model']} "
        f"bs={config['globalSequences']} phase=post_decay_eval epoch={epoch} "
        f"wd={config['lockedWd']} source={endpoint} output={output}",
        flush=True,
    )
    subprocess.run(
        ["python", adaptive.TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True
    )
    log_path = state_dir(config) / "logs" / f"post_decay_eval_e{epoch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={config['nprocPerNode']}",
                adaptive.TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )
    return locked.parse_evaluation(
        log_path, epoch, phase="post_decay", checkpoint=endpoint
    )


def run_postdecay(
    config: dict[str, Any], epoch: int, source: Path
) -> dict[str, Any]:
    path = post_result_path(config, epoch)
    if path.is_file():
        result = json.loads(path.read_text())
        emit_result(config, result)
        return result
    batch = int(config["globalSequences"])
    expected_source = checkpoint_for(config, epoch)
    if source != expected_source or not source.is_dir():
        raise FileNotFoundError(f"missing exact POST source checkpoint {expected_source}")
    output = state_dir(config) / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{adaptive.total_step(epoch, batch)}"
    if endpoint.is_dir():
        result = evaluate_endpoint(config, epoch, endpoint)
        source_kind = "recovered_post_decay_endpoint"
    else:
        name = run_name(config, "postdecay", epoch)
        arguments = locked.postdecay_training_arguments(
            config, epoch, source, output, name
        )
        arguments = locked.replace_argument(
            arguments,
            "--trainer.callbacks.wandb.tags=",
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,post-decay,locked-wd,post-only-policy]",
        )
        print(
            f"SMALL_PREDECAY_POLICY_STAGE_START model={config['model']} "
            f"bs={config['globalSequences']} phase=post_decay_train epoch={epoch} "
            f"wd={config['lockedWd']} source={source} output={output}",
            flush=True,
        )
        subprocess.run(
            ["python", adaptive.TRAINING_SCRIPT, name, "--dry-run", *arguments],
            check=True,
        )
        log_path = state_dir(config) / "logs" / f"post_decay_train_e{epoch}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log_file:
            common.stream_command(
                [
                    "torchrun",
                    f"--nproc-per-node={config['nprocPerNode']}",
                    adaptive.TRAINING_SCRIPT,
                    name,
                    *arguments,
                ],
                log_file,
            )
        result = common.parse_stage_result(log_path, epoch)
        source_kind = "post_decay_training"
    result.update(
        {
            "epoch": epoch,
            "phase": "post_decay",
            "comparisonGroup": "post_decay",
            "wd": str(config["lockedWd"]),
            "variant": "DR+WT+EmbedWD",
            "policy": POLICY,
            "sourcePreDecayCheckpoint": str(source),
            "checkpoint": str(endpoint),
            "source": source_kind,
        }
    )
    locked.atomic_json(path, result)
    emit_result(config, result)
    return result


def saturated(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    return Decimal(str(current["validationExact"])) >= Decimal(
        str(previous["validationExact"])
    )


def decision(
    config: dict[str, Any], results: dict[int, dict[str, Any]], epoch: int
) -> dict[str, Any]:
    sources = sorted(results)[-POST_DECAY_SOURCE_COUNT:]
    if len(sources) != POST_DECAY_SOURCE_COUNT or sources[-1] != epoch:
        raise RuntimeError("POST saturation requires the latest three results")
    selected_epoch = min(
        sources,
        key=lambda candidate: (
            Decimal(str(results[candidate]["validationExact"])),
            candidate,
        ),
    )
    selected = results[selected_epoch]
    return {
        "status": "complete",
        "policy": POLICY,
        "lockedWd": str(config["lockedWd"]),
        "terminationReason": "post_decay_non_improvement",
        "evaluatedThroughEpoch": epoch,
        "saturationEpoch": epoch,
        "postDecayDecisionGroup": "post_decay",
        "postDecaySelectionGroup": "post_decay",
        "postDecaySaturationCriterion": "strict_non_improvement",
        "postDecaySaturated": True,
        "postDecaySourceEpochs": sources,
        "postDecayValidationExact": {
            str(candidate): float(results[candidate]["validationExact"])
            for candidate in sources
        },
        "selectedPostDecayEpoch": selected_epoch,
        "selectedPostDecayValidationExact": float(selected["validationExact"]),
        "selectedCheckpoint": selected["checkpoint"],
        "preserveExistingSelection": False,
    }


def emit_complete(config: dict[str, Any], value: dict[str, Any]) -> None:
    print(
        f"SMALL_POSTDECAY_FINALIZER_COMPLETE model={config['model']} "
        f"bs={config['globalSequences']} status=complete "
        f"json={json.dumps(value, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    state_dir(config).mkdir(parents=True, exist_ok=True)
    if selection_path(config).is_file():
        value = json.loads(selection_path(config).read_text())
        emit_complete(config, value)
        return value
    results = prior_results(config)
    previous_epoch = int(config["resumeEpoch"])
    while True:
        epoch = next_epoch(config, previous_epoch)
        checkpoint = train_frontier(config, previous_epoch, epoch)
        current = run_postdecay(config, epoch, checkpoint)
        previous = results[previous_epoch]
        results[epoch] = current
        if saturated(previous, current):
            value = decision(config, results, epoch)
            locked.atomic_json(selection_path(config), value)
            emit_complete(config, value)
            return value
        print(
            f"SMALL_POSTDECAY_ONLY_CONTINUE model={config['model']} "
            f"bs={config['globalSequences']} epoch={epoch} wd={config['lockedWd']} "
            f"next_epoch={next_epoch(config, epoch)}",
            flush=True,
        )
        previous_epoch = epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    validate_config(config)
    if args.validate_only:
        print(f"validated {args.manifest}")
        return
    run(config)


if __name__ == "__main__":
    main()
