#!/usr/bin/env python3
"""Submit one Dense-1B DR+WT E1->E24 chain as a single Beaker experiment.

Every stage performs a complete WSD run and held-out evaluation. The following
stage resumes the preceding stage's exact pre-decay checkpoint, never its decayed
endpoint. E1 starts fresh with ordinary packing; dynamic repacking begins at E2.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import submit_dense_step1_data_loader_coordinate as endpoint

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
TARGETS = (1, 2, 4, 8, 12, 16, 20, 24)
SATURATION_TARGETS = (1, 2, 4, *range(8, 65, 4))
REVISION = "sewonm/icsl"
OUTPUT_ROOT = (
    "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/" "wt_sequential_e24_r01"
)
COORDINATES = {
    (64, "0.3"): {
        "runId": "drwt64-lr1e-3-wd0.3",
        "sourceExperiment": "01KZDCD9AYJ0REGZTE2YG18NHF",
        "nodes": 1,
    },
    (512, "0.333"): {
        "runId": "drwt512-lr1e-3-wd0.333",
        "sourceExperiment": "01KZFVTS3EHFR7HQ5ZY9DAF9NS",
        "nodes": 4,
    },
    (512, "1.0"): {
        "runId": "drwt512-lr1e-3-wd1.0",
        "sourceExperiment": "01KZGE4HMCEWMBP86YEAGSP1B7",
        "nodes": 4,
    },
}


def coordinate_run_id(args: argparse.Namespace) -> str:
    prefix = "drwtembwd" if args.decay_embeddings else "drwt"
    mlp = ""
    if getattr(args, "mlp_weight_decay", None) is not None:
        mlp = f"-mlp-{args.mlp_weight_decay_scope}-wd{args.mlp_weight_decay}"
    return f"{prefix}{args.global_sequences}-lr1e-3-wd{args.weight_decay}{mlp}"


def coordinate_method(args: argparse.Namespace) -> str:
    method = f"drwtembwd{args.global_sequences}"
    if getattr(args, "mlp_weight_decay", None) is not None:
        method += "mlpupperwd1"
    return method


def sequential_run_id(args: argparse.Namespace) -> str:
    mode = "wtseq-embwd" if args.decay_embeddings else "wtseq"
    mlp = ""
    if getattr(args, "mlp_weight_decay", None) is not None:
        mlp = f"-mlp-{args.mlp_weight_decay_scope}-wd{args.mlp_weight_decay}"
    return f"{mode}-bs{args.global_sequences}-wd{args.weight_decay}{mlp}"


def registry_key(args: argparse.Namespace) -> str:
    if getattr(args, "mlp_weight_decay", None) is not None:
        return "weightTyingMlpDecaySequentialRuns"
    return (
        "weightTyingEmbeddingDecaySequentialRuns"
        if args.decay_embeddings
        else "weightTyingSequentialRuns"
    )


def targets_for(args: argparse.Namespace) -> tuple[int, ...]:
    return SATURATION_TARGETS if getattr(args, "mlp_weight_decay", None) is not None else TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-sequences", type=int, choices=(64, 512), required=True)
    parser.add_argument("--weight-decay", choices=("0.3", "0.333", "1.0"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workspace", default=endpoint.FLEX2_WORKSPACE)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="replace this coordinate's failed registry entry and reuse its guarded output path",
    )
    parser.add_argument(
        "--resume-from-predecay-epoch",
        type=int,
        choices=TARGETS,
        help=(
            "for a failed retry, resume the first incomplete stage from its retained exact "
            "pre-decay checkpoint instead of replaying that stage's stable training segment"
        ),
    )
    parser.add_argument(
        "--canonical-output",
        action="store_true",
        help="write directly to the canonical bs{batch}_dr_wt_lr{lr}_wd{wd} trajectory",
    )
    parser.add_argument(
        "--decay-embeddings",
        action="store_true",
        help="Apply global WD to the tied embeddings/LM-head matrix.",
    )
    parser.add_argument(
        "--stop-on-saturation",
        action="store_true",
        help="Stop the persistent chain when held-out CE fails to strictly improve.",
    )
    parser.add_argument("--mlp-weight-decay", choices=("1.0",))
    parser.add_argument(
        "--mlp-weight-decay-scope",
        choices=("all", "upper-half", "upper-half-w2"),
        default="all",
    )
    args = parser.parse_args()
    if args.stop_on_saturation and not args.decay_embeddings:
        parser.error("--stop-on-saturation is currently scoped to --decay-embeddings")
    if args.resume_from_predecay_epoch is not None and not args.retry_failed:
        parser.error("--resume-from-predecay-epoch requires --retry-failed")
    if args.mlp_weight_decay is not None:
        if not args.decay_embeddings:
            parser.error("--mlp-weight-decay requires --decay-embeddings")
        if args.mlp_weight_decay_scope != "upper-half":
            parser.error("this study requires --mlp-weight-decay-scope=upper-half")
        if not args.stop_on_saturation:
            parser.error("the MLP WD study requires --stop-on-saturation")
        if (args.global_sequences, args.weight_decay) not in {
            (64, "0.3"),
            (512, "0.333"),
        }:
            parser.error("the MLP WD study has only BS64/WD0.3 and BS512/WD0.333")
    if args.decay_embeddings:
        args.canonical_output = True
    if (args.global_sequences, args.weight_decay) not in COORDINATES:
        parser.error("requested coordinate is not one of the three authorized DR+WT chains")
    if args.revision != REVISION:
        parser.error(f"the sequential study must use the fetchable revision {REVISION}")
    if args.workspace != endpoint.FLEX2_WORKSPACE:
        parser.error(f"the sequential study submits only to {endpoint.FLEX2_WORKSPACE}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.name):
        parser.error("--name must be a lowercase Beaker name component")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.suffix):
        parser.error("--suffix must be a lowercase run-name component")
    return args


def stage_namespace(args: argparse.Namespace, epoch: int, source_checkpoint: str) -> Any:
    coordinate = COORDINATES[(args.global_sequences, args.weight_decay)]
    return SimpleNamespace(
        method="dynamic_repacking",
        global_sequences=args.global_sequences,
        target_epoch=epoch,
        learning_rate="1e-3",
        weight_decay=args.weight_decay,
        source_experiment=coordinate["sourceExperiment"],
        source_checkpoint=source_checkpoint,
        revision=args.revision,
        name=args.name,
        suffix=args.suffix,
        weight_tying=True,
        decay_embeddings=args.decay_embeddings,
        mlp_weight_decay=getattr(args, "mlp_weight_decay", None),
        mlp_weight_decay_scope=getattr(args, "mlp_weight_decay_scope", "all"),
        workspace=args.workspace,
        priority=args.priority,
        register=False,
        print_only=False,
        allow_pending_flex2_jobs=True,
    )


def with_stage_log_capture(shell: str, output: str, epoch: int) -> tuple[str, str]:
    """Capture global-rank-zero output without making every replica write the same file."""
    log_path = f"{output}/.embwd_e{epoch}.log"
    lines = shell.splitlines()
    torchrun_indexes = [index for index, line in enumerate(lines) if line.startswith("torchrun ")]
    if len(torchrun_indexes) != 1:
        raise RuntimeError(f"expected exactly one torchrun command for E{epoch}")
    index = torchrun_indexes[0]
    torchrun = lines[index]
    lines[index : index + 1] = [
        'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then',
        f"mkdir -p {shlex.quote(output)}",
        f"{torchrun} 2>&1 | tee {shlex.quote(log_path)}",
        "else",
        torchrun,
        "fi",
    ]
    return "\n".join(lines), log_path


def saturation_gate(output: str, epoch: int, previous_epoch: int | None, log_path: str) -> list[str]:
    decision = f"{output}/.embwd_e{epoch}.decision"
    validation = f"{output}/.embwd_e{epoch}.validation"
    previous_validation = (
        None if previous_epoch is None else f"{output}/.embwd_e{previous_epoch}.validation"
    )
    code = "".join(
        [
        "import os,pathlib,re\n",
        f"log=pathlib.Path({log_path!r})\n",
        "text=re.sub(r'\\x1b\\[[0-?]*[ -/]*[@-~]','',log.read_text())\n",
        "vals=re.findall(r'wandb:\\s+eval/heldout/dclm-validation-0802/CE loss\\s+([0-9]+(?:\\.[0-9]+)?)',text)\n",
        "if not vals: vals=re.findall(r'dclm-validation-0802/CE loss=([0-9]+(?:\\.[0-9]+)?)\\s*$',text,re.M)\n",
        "assert vals, 'missing held-out validation CE'\n",
        "value=float(vals[-1])\n",
        ]
        + [
            "previous=None\n"
            if previous_validation is None
            else f"previous=float(pathlib.Path({previous_validation!r}).read_text())\n"
        ]
        + [
        f"v=pathlib.Path({validation!r}); vt=v.with_suffix(v.suffix+'.tmp'); vt.write_text(str(value)); os.replace(vt,v)\n",
        "action='continue' if previous is None or value < previous else 'stop'\n",
        f"d=pathlib.Path({decision!r}); dt=d.with_suffix(d.suffix+'.tmp'); dt.write_text(action); os.replace(dt,d)\n",
        f"print(f'SEQUENTIAL_EMBWD_DECISION epoch={epoch} validation={{value}} previous={{previous}} action={{action}}')"
        ]
    )
    return [
        'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then',
        f"python -c {shlex.quote(code)}",
        "fi",
        f"for _wait in $(seq 1 600); do test -f {shlex.quote(decision)} && break; sleep 1; done",
        f"test -f {shlex.quote(decision)}",
        f'if [ "$(cat {shlex.quote(decision)})" = stop ]; then',
        f'echo "SEQUENTIAL_EMBWD_SATURATED epoch={epoch}"',
        "exit 0",
        "fi",
    ]


def build_chain(args: argparse.Namespace) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    coordinate = COORDINATES[(args.global_sequences, args.weight_decay)]
    base = endpoint.beaker_spec(coordinate["sourceExperiment"])
    first_args = stage_namespace(args, 1, "fresh")
    training_script, base_arguments = endpoint.audit_source_spec(base, first_args)

    old_root = endpoint.MODEL_ROOT
    endpoint.MODEL_ROOT = old_root if (args.canonical_output or args.decay_embeddings) else OUTPUT_ROOT
    targets = targets_for(args)
    try:
        stage_specs = []
        stage_records = []
        active_targets = []
        output = ""
        previous_epoch = None
        resume_after_epoch = int(getattr(args, "resume_after_epoch", 0))
        resume_from_predecay_epoch = getattr(args, "resume_from_predecay_epoch", None)
        for target in targets:
            source = (
                "fresh"
                if previous_epoch is None
                else f"{output}/step{endpoint.stable_step(previous_epoch, args.global_sequences)}"
            )
            if target == resume_from_predecay_epoch:
                source = f"{output}/step{endpoint.stable_step(target, args.global_sequences)}"
            current_args = stage_namespace(args, target, source)
            stage_spec, output = endpoint.build_spec(
                copy.deepcopy(base), current_args, training_script, base_arguments
            )
            if target == resume_from_predecay_epoch:
                shell = stage_spec["tasks"][0]["arguments"][2]
                checkpoint_guard = f"test '!' -e {source}\n"
                if checkpoint_guard not in shell:
                    raise RuntimeError(
                        f"E{target} pre-decay recovery is missing the expected checkpoint guard"
                    )
                stage_spec["tasks"][0]["arguments"][2] = shell.replace(
                    checkpoint_guard, "", 1
                )
            if target == 1 and args.retry_failed:
                shell = stage_spec["tasks"][0]["arguments"][2]
                output_guard = f"test '!' -e {output}\n"
                if output_guard not in shell:
                    raise RuntimeError("failed retry spec is missing the expected output-root guard")
                stage_spec["tasks"][0]["arguments"][2] = shell.replace(
                    output_guard, "", 1
                )
            if target > resume_after_epoch:
                stage_specs.append(stage_spec)
                active_targets.append(target)
            stage_records.append(
                {
                    "epoch": target,
                    "status": "planned",
                    "sourceCheckpoint": source,
                    "preDecayCheckpoint": (
                        f"{output}/step{endpoint.stable_step(target, args.global_sequences)}"
                    ),
                    "endpointCheckpoint": (
                        f"{output}/step{endpoint.total_step(target, args.global_sequences)}"
                    ),
                }
            )
            previous_epoch = target
    finally:
        endpoint.MODEL_ROOT = old_root

    if not active_targets:
        raise RuntimeError("failed retry has no remaining sequential stage")
    spec = stage_specs[0]
    task = spec["tasks"][0]
    commands = ["set -euo pipefail"]
    previous_target = resume_after_epoch or None
    if args.stop_on_saturation and previous_target is not None:
        previous_validation = getattr(args, "resume_validation", None)
        if previous_validation is None:
            raise RuntimeError("EmbedWD retry is missing the completed frontier validation CE")
        validation_path = f"{output}/.embwd_e{previous_target}.validation"
        seed_code = (
            "import os,pathlib\n"
            f"p=pathlib.Path({validation_path!r})\n"
            "p.parent.mkdir(parents=True,exist_ok=True)\n"
            f"t=p.with_suffix(p.suffix+'.tmp'); t.write_text({str(previous_validation)!r}); "
            "os.replace(t,p)"
        )
        commands.extend(
            [
                'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then',
                f"python -c {shlex.quote(seed_code)}",
                "fi",
            ]
        )
    for target, stage_spec in zip(active_targets, stage_specs):
        stage_shell = stage_spec["tasks"][0]["arguments"][2]
        stage_shell = stage_shell.removeprefix("set -euo pipefail\n")
        log_path = ""
        if args.stop_on_saturation:
            stage_shell, log_path = with_stage_log_capture(stage_shell, output, target)
        commands.extend(
            [
                f'echo "SEQUENTIAL_WT_STAGE_START epoch={target}"',
                stage_shell,
                f'echo "SEQUENTIAL_WT_STAGE_COMPLETE epoch={target}"',
            ]
        )
        if args.stop_on_saturation and target != targets[-1]:
            commands.extend(saturation_gate(output, target, previous_target, log_path))
        previous_target = target
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    task["replicas"] = coordinate["nodes"]
    task["resources"] = {"gpuCount": endpoint.GPUS_PER_NODE, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": "0s", "autoResume": False}
    spec.pop("description", None)
    return spec, output, stage_records


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def register_submission(
    args: argparse.Namespace,
    experiment: str,
    output: str,
    stages: list[dict[str, Any]],
) -> None:
    report = json.loads(REPORT_PATH.read_text())
    coordinate = COORDINATES[(args.global_sequences, args.weight_decay)]
    records = report.setdefault(registry_key(args), [])
    run_id = sequential_run_id(args)
    matching = [record for record in records if record.get("id") == run_id]
    if args.retry_failed:
        if len(matching) != 1 or matching[0].get("status") not in {"failed", "canceled"}:
            raise RuntimeError(
                f"--retry-failed requires exactly one failed or canceled entry for {run_id}"
            )
        previous = copy.deepcopy(matching[0])
        previous_stages = {
            int(stage["epoch"]): stage
            for stage in previous.get("stages", [])
            if stage.get("status") == "complete"
        }
        attempts = list(previous.pop("attempts", []))
        attempts.append(previous)
        record = matching[0]
        record.clear()
        record["attempts"] = attempts
    else:
        if matching:
            raise RuntimeError(f"duplicate sequential registry entry for {run_id}")
        record = {}
        records.append(record)
        previous_stages = {}
    stages = [previous_stages.get(int(stage["epoch"]), stage) for stage in stages]
    current_epoch = next(
        (int(stage["epoch"]) for stage in stages if stage.get("status") != "complete"), None
    )
    record.update(
        {
            "id": run_id,
            "coordinateRunId": coordinate_run_id(args),
            "batchSequences": args.global_sequences,
            "lr": "1e-3",
            "wd": args.weight_decay,
            "weightTying": True,
            "decayEmbeddings": args.decay_embeddings,
            "mlpWeightDecay": getattr(args, "mlp_weight_decay", None),
            "mlpWeightDecayScope": (
                getattr(args, "mlp_weight_decay_scope", None)
                if getattr(args, "mlp_weight_decay", None) is not None
                else None
            ),
            "dataOrder": "dynamic_repacking_from_e2",
            "status": "submitted",
            "currentEpoch": current_epoch,
            "targets": list(targets_for(args)),
            "experiment": experiment,
            "beaker": experiment,
            "revision": args.revision,
            "output": output,
            "nodeCount": coordinate["nodes"],
            "gpuCount": coordinate["nodes"] * endpoint.GPUS_PER_NODE,
            "stages": stages,
            "reason": (
                (
                    f"Retried after a failed persistent attempt from E{current_epoch}. "
                    if args.retry_failed
                    else ""
                )
                + "Submitted as one persistent saturation-gated job. Every stage evaluates its WSD "
                "decayed endpoint and the next stage resumes the preceding exact pre-decay "
                "checkpoint; dynamic repacking begins at E2. "
                + (
                    "Outputs use the canonical coordinate trajectory directory."
                    if args.canonical_output or args.decay_embeddings
                    else "Outputs require a guarded canonical merge after completion."
                )
            ),
            "canonicalOutput": endpoint.trajectory_output(
                stage_namespace(args, targets_for(args)[-1], "unused")
            ),
            "mergeAfterCompletion": not (args.canonical_output or args.decay_embeddings),
            "stopOnSaturation": args.stop_on_saturation,
        }
    )
    if args.decay_embeddings:
        coordinate_id = coordinate_run_id(args)
        coordinate_matches = [
            run for run in report.get("runs", []) if run.get("id") == coordinate_id
        ]
        if coordinate_matches:
            coordinate_run = coordinate_matches[0]
        else:
            coordinate_run = {
                "id": coordinate_id,
                "method": coordinate_method(args),
                "batchSequences": args.global_sequences,
                "dataOrder": "dynamic_repacking",
                "weightTying": True,
                "decayEmbeddings": True,
                "mlpWeightDecay": getattr(args, "mlp_weight_decay", None),
                "mlpWeightDecayScope": (
                    getattr(args, "mlp_weight_decay_scope", None)
                    if getattr(args, "mlp_weight_decay", None) is not None
                    else None
                ),
                "lr": "1e-3",
                "wd": args.weight_decay,
                "attemptedEpochs": list(targets_for(args)),
                "sourceExperiment": coordinate["sourceExperiment"],
                "sourceCheckpoint": "fresh",
                "gpuCount": coordinate["nodes"] * endpoint.GPUS_PER_NODE,
                "nodeCount": coordinate["nodes"],
                "rankMicrobatchSequences": endpoint.RANK_MICROBATCH_SEQUENCES,
                "plannedTargets": list(targets_for(args)),
                "results": {},
            }
            report.setdefault("runs", []).append(coordinate_run)
        coordinate_run.update(
            {
                "status": "submitted",
                "activeEpoch": current_epoch,
                "experiment": experiment,
                "beaker": experiment,
                "revision": args.revision,
                "output": output,
                "reason": (
                    "Persistent DR+WT chain with global WD on the tied "
                    "embeddings/LM-head matrix"
                    + (
                        f" and WD {args.mlp_weight_decay} on upper-half MLP w1/w2/w3 matrices."
                        if getattr(args, "mlp_weight_decay", None) is not None
                        else "."
                    )
                ),
            }
        )
        column_key = coordinate_method(args)
        if not any(
            column.get("key") == column_key for column in report.get("columns", [])
        ):
            report.setdefault("columns", []).append(
                {
                    "key": column_key,
                    "label": (
                        f"BS{args.global_sequences} · DR+WT+EmbedWD"
                        + (
                            "+UpperMLPWD"
                            if getattr(args, "mlp_weight_decay", None) is not None
                            else ""
                        )
                    ),
                    "batchSequences": args.global_sequences,
                    "dataOrder": "dynamic_repacking",
                    "weightTying": True,
                    "decayEmbeddings": True,
                    "mlpWeightDecay": getattr(args, "mlp_weight_decay", None),
                    "mlpWeightDecayScope": (
                        getattr(args, "mlp_weight_decay_scope", None)
                        if getattr(args, "mlp_weight_decay", None) is not None
                        else None
                    ),
                    "initialWd": args.weight_decay,
                    "color": "#7c3aed" if args.global_sequences == 64 else "#4f46e5",
                }
            )
    write_report(report)


def validate_registration(args: argparse.Namespace) -> None:
    if not args.register:
        return
    report = json.loads(REPORT_PATH.read_text())
    run_id = sequential_run_id(args)
    matching = [
        record
        for record in report.get(registry_key(args), [])
        if record.get("id") == run_id
    ]
    if args.retry_failed:
        if len(matching) != 1 or matching[0].get("status") not in {"failed", "canceled"}:
            raise RuntimeError(
                f"--retry-failed requires exactly one failed or canceled entry for {run_id}"
            )
    elif matching:
        raise RuntimeError(f"duplicate sequential registry entry for {run_id}")


def configure_failed_retry(args: argparse.Namespace) -> None:
    args.resume_after_epoch = 0
    args.resume_validation = None
    if not args.retry_failed:
        return
    report = json.loads(REPORT_PATH.read_text())
    run_id = sequential_run_id(args)
    record = next(
        record
        for record in report.get(registry_key(args), [])
        if record.get("id") == run_id
    )
    stages = {int(stage["epoch"]): stage for stage in record.get("stages", [])}
    completed_frontier = 0
    targets = targets_for(args)
    for target in targets:
        if stages.get(target, {}).get("status") != "complete":
            break
        completed_frontier = target
    if completed_frontier == 0:
        raise RuntimeError("failed retry has no validated completed frontier to resume")
    args.resume_after_epoch = completed_frontier
    args.resume_validation = stages[completed_frontier].get("validationExact") or stages[
        completed_frontier
    ].get("validation")
    if args.decay_embeddings and args.resume_validation is None:
        raise RuntimeError("EmbedWD retry frontier is missing held-out validation CE")
    resume_from_predecay_epoch = getattr(args, "resume_from_predecay_epoch", None)
    if resume_from_predecay_epoch is not None:
        first_incomplete = next(target for target in targets if target > completed_frontier)
        if resume_from_predecay_epoch != first_incomplete:
            raise RuntimeError(
                "--resume-from-predecay-epoch must equal the first incomplete stage "
                f"E{first_incomplete}"
            )
        expected = endpoint.stable_step(first_incomplete, args.global_sequences)
        checkpoint = stages[first_incomplete].get("preDecayCheckpoint", "")
        if not checkpoint.endswith(f"/step{expected}"):
            raise RuntimeError(
                f"registered E{first_incomplete} pre-decay checkpoint does not end in step{expected}"
            )


def main() -> None:
    args = parse_args()
    validate_registration(args)
    configure_failed_retry(args)
    spec, output, stages = build_chain(args)
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
        register_submission(args, ids[0], output, stages)


if __name__ == "__main__":
    main()
