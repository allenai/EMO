#!/usr/bin/env python3
"""Refresh small-dense chains after the locked-WD pre-decay transition."""

from __future__ import annotations

import json
import re
import statistics
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import monitor_small_dense_dr_wt_embedwd_chains as legacy

REPORTS = legacy.REPORTS
POLICY = "locked_wd_predecay_saturation_v1"
FINALIZER_POLICY = "locked_wd_requested_postdecay_finalizer_v1"
POLICIES = {POLICY, FINALIZER_POLICY}
RESULT = re.compile(
    r"SMALL_PREDECAY_POLICY_RESULT model=(153m|474m) bs=([0-9]+) "
    r"phase=(pre_decay|post_decay) epoch=([0-9]+) wd=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
STAGE_START = re.compile(
    r"SMALL_PREDECAY_POLICY_STAGE_START model=(153m|474m) bs=([0-9]+) "
    r"phase=([^ ]+) epoch=([0-9]+) wd=([^ ]+) source=([^ ]+) output=([^\s]+)"
)
SATURATED = re.compile(
    r"SMALL_PREDECAY_POLICY_SATURATED model=(153m|474m) bs=([0-9]+) "
    r"epoch=([0-9]+) wd=([^ ]+) comparison_group=pre_decay"
)
SELECTED = re.compile(
    r"SMALL_PREDECAY_POLICY_SELECTED model=(153m|474m) bs=([0-9]+) "
    r"saturation_epoch=([0-9]+) wd=([^ ]+) selected_epoch=([0-9]+) "
    r"validation=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
CONTINUED = re.compile(
    r"SMALL_PREDECAY_POLICY_CONTINUE model=(153m|474m) bs=([0-9]+) "
    r"provisional_epoch=([0-9]+) wd=([^ ]+) comparison_group=post_decay "
    r"next_epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
FINALIZED = re.compile(
    r"SMALL_POSTDECAY_FINALIZER_COMPLETE model=(153m|474m) bs=([0-9]+) "
    r"status=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
POST_DECAY_SATURATION_CRITERION = "strict_non_improvement"


def parsed_results(
    logs: str, model: str, batch: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pre: dict[str, dict[str, Any]] = {}
    post: dict[str, dict[str, Any]] = {}
    for parsed_model, parsed_batch, phase, epoch, wd, payload in RESULT.findall(logs):
        if parsed_model != model or int(parsed_batch) != batch:
            continue
        result = json.loads(payload)
        if str(result.get("wd")) != str(wd):
            raise RuntimeError(f"{model} BS{batch} E{epoch}: result WD mismatch")
        target = pre if phase == "pre_decay" else post
        target[str(int(epoch))] = result
    return pre, post


def is_scheduled_frontier(record: dict[str, Any], epoch: int) -> bool:
    initial = [int(value) for value in record["initialTargets"]]
    if epoch in initial:
        return True
    return epoch > initial[-1] and (epoch - initial[-1]) % int(record["epochIncrement"]) == 0


def is_policy_predecay_frontier(record: dict[str, Any], epoch: int) -> bool:
    return epoch >= int(
        record.get("historicalPreDecayStartEpoch", 1)
    ) and is_scheduled_frontier(record, epoch)


def retain_scheduled_predecay_results(record: dict[str, Any]) -> None:
    results = record.get("preDecayResults", {})
    excluded = {
        epoch: result
        for epoch, result in results.items()
        if not is_policy_predecay_frontier(record, int(epoch))
    }
    if excluded:
        record.setdefault("excludedPreDecayResults", {}).update(excluded)
        record["excludedPreDecayReason"] = (
            "Before the E8 historical PD start or not a configured legacy WSD frontier; "
            "excluded from pre-decay saturation."
        )
    record["preDecayResults"] = {
        epoch: result
        for epoch, result in results.items()
        if is_policy_predecay_frontier(record, int(epoch))
    }


def extend_report_display_epochs(report: dict[str, Any]) -> None:
    """Keep report tables aligned with every completed locked-policy frontier."""
    observed_by_batch: dict[str, set[int]] = {}
    for record in report.get("adaptiveDrWtEmbedWdChains", []):
        if record.get("policy") not in POLICIES:
            continue
        batch = str(int(record["batchSequences"]))
        observed = observed_by_batch.setdefault(batch, set())
        for key in ("preDecayResults", "postDecayResults"):
            observed.update(int(epoch) for epoch in record.get(key, {}))

    all_observed = set().union(*observed_by_batch.values()) if observed_by_batch else set()
    report["targetEpochs"] = sorted(
        {*report.get("targetEpochs", []), *all_observed}, key=float
    )
    batch_targets = report.setdefault("batchTargetEpochs", {})
    for batch, observed in observed_by_batch.items():
        batch_targets[batch] = sorted(
            {*batch_targets.get(batch, []), *observed}, key=float
        )


def selection_postdecay_saturated(selection: dict[str, Any]) -> bool:
    sources = [int(epoch) for epoch in selection.get("postDecaySourceEpochs", [])]
    values = selection.get("postDecayValidationExact") or {}
    if len(sources) != 3 or any(str(epoch) not in values for epoch in sources):
        raise RuntimeError("post-decay decision requires exactly three ordered results")
    previous = float(values[str(sources[-2])])
    current = float(values[str(sources[-1])])
    return current >= previous


def policy_health(logs: str, record: dict[str, Any], state: str) -> dict[str, Any] | None:
    starts = list(STAGE_START.finditer(logs))
    if not starts:
        return None
    start = starts[-1]
    model, batch, phase, epoch, wd, source, output = start.groups()
    segment = logs[start.start() :]
    run_ids = legacy.WANDB_RUN.findall(segment)
    run_id = run_ids[-1] if run_ids else None
    save_folders = legacy.SAVE_FOLDER.findall(segment)
    save_folder = legacy.normalize_log_value(save_folders[-1] if save_folders else None)
    load_paths = legacy.LOAD_CHECKPOINT.findall(segment)
    load_path = legacy.normalize_log_value(load_paths[-1] if load_paths else None)
    steps, losses, nonfinite, last_timestamp, total_steps = legacy.parse_metric_samples(segment)

    critical: list[str] = []
    warnings: list[str] = []
    model_root = str(Path(str(record["outputByWd"][str(record["lockedWd"])])).parent)
    if str(wd) != str(record["lockedWd"]):
        critical.append("locked-wd-mismatch")
    if not output.startswith(model_root.rstrip("/") + "/"):
        critical.append("stage-output-path-mismatch")
    if not source.startswith(model_root.rstrip("/") + "/"):
        critical.append("checkpoint-source-path-mismatch")
    if save_folder and save_folder != output:
        critical.append("trainer-save-path-mismatch")
    if load_path and load_path != source:
        critical.append("checkpoint-load-path-mismatch")
    if load_path is None:
        critical.append("missing-resume-checkpoint-load")
    regressions = [(left, right) for left, right in pairwise(steps) if right < left]
    if regressions:
        critical.append("wandb-step-regression")
    if nonfinite:
        critical.append("nonfinite-training-loss")

    recent_median: float | None = None
    baseline_median: float | None = None
    spike_count = 0
    maximum: float | None = None
    if losses:
        maximum = max(losses)
        recent = losses[-50:]
        baseline_pool = losses[-250:-50] or losses[:-50] or losses
        recent_median = statistics.median(recent)
        baseline_median = statistics.median(baseline_pool)
        spike_count = sum(value > baseline_median + 0.75 for value in losses)
        if len(recent) >= 25 and recent_median > baseline_median + 0.5:
            critical.append("sustained-training-loss-shift")
        elif spike_count:
            warnings.append("isolated-finite-loss-spikes")

    stalled_minutes: float | None = None
    training_phase = phase in {"constant_train", "post_decay_train"}
    if (
        training_phase
        and last_timestamp
        and state == "running"
        and steps
        and total_steps
        and steps[-1] < total_steps
    ):
        last_metric = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
        stalled_minutes = (datetime.now(tz=UTC) - last_metric).total_seconds() / 60
        if stalled_minutes > 45:
            critical.append("training-metrics-stalled")
    if steps and run_id is None:
        warnings.append("wandb-run-id-not-found-in-job-log")
    status = "critical" if critical else "warning" if warnings else "healthy"
    return {
        "status": status,
        "checkedAt": datetime.now(tz=UTC).isoformat(),
        "run": run_id,
        "url": (f"https://wandb.ai/ai2-llm/sewonm-icsl/runs/{run_id}" if run_id else None),
        "model": model,
        "batchSequences": int(batch),
        "phase": phase,
        "epoch": int(epoch),
        "wd": wd,
        "sourceCheckpoint": source,
        "stageOutput": output,
        "saveFolder": save_folder,
        "loadPath": load_path,
        "samples": len(losses),
        "firstStep": steps[0] if steps else None,
        "latestStep": steps[-1] if steps else None,
        "totalSteps": total_steps,
        "recentLossMedian": round(recent_median, 6) if recent_median is not None else None,
        "baselineLossMedian": round(baseline_median, 6) if baseline_median is not None else None,
        "maxLoss": maximum,
        "isolatedSpikeCount": spike_count,
        "stepRegressions": regressions[:5],
        "nonfiniteLosses": nonfinite[:5],
        "stalledMinutes": round(stalled_minutes, 1) if stalled_minutes is not None else None,
        "warnings": warnings,
        "criticalSignals": critical,
        "shouldRecover": bool(critical),
        "automaticPathRecoveryReady": False,
    }


def update_policy_record(model: str, record: dict[str, Any]) -> str:
    batch = int(record["batchSequences"])
    experiment_id = str(record["experiment"])
    if record.get("manualNoWorkFinalization"):
        pre_metrics = ", ".join(
            f"[PD] E{epoch} CE{result.get('validationExact', result.get('validation'))}"
            for epoch, result in sorted(
                record.get("preDecayResults", {}).items(),
                key=lambda item: int(item[0]),
            )
        )
        post_metrics = ", ".join(
            f"[POST] E{epoch} CE{result.get('validationExact', result.get('validation'))}"
            for epoch, result in sorted(
                record.get("postDecayResults", {}).items(),
                key=lambda item: int(item[0]),
            )
        )
        return (
            f"{model} BS{batch} complete; locked WD{record['lockedWd']}; "
            f"{pre_metrics or '[PD] provenance only'}; "
            f"{post_metrics or '[POST] not available'}; phase=done; {experiment_id}"
        )
    inspected = legacy.inspect_experiment(experiment_id)
    state = legacy.experiment_state(inspected)
    jobs = [job for job in inspected.get("jobs") or [] if job.get("id")]
    record["status"] = state
    record["beakerStatus"] = state
    if jobs:
        record["job"] = jobs[-1]["id"]
        record["jobs"] = [job["id"] for job in jobs]
    logs = ""
    if state in {"running", "complete", "failed"}:
        logs = legacy.ANSI.sub("", legacy.run(["beaker", "experiment", "logs", experiment_id]))
    pre, post = parsed_results(logs, model, batch)
    pre = {
        epoch: result
        for epoch, result in pre.items()
        if is_policy_predecay_frontier(record, int(epoch))
    }
    if pre:
        record.setdefault("preDecayResults", {}).update(pre)
    if post:
        record.setdefault("postDecayResults", {}).update(post)
    retain_scheduled_predecay_results(record)
    record["postDecaySaturationCriterion"] = POST_DECAY_SATURATION_CRITERION
    existing_selection = record.get("postDecaySelection")
    if existing_selection:
        existing_selection.setdefault("postDecayDecisionGroup", "post_decay")
        existing_selection.setdefault("postDecaySelectionGroup", "post_decay")

    start_matches = [
        match
        for match in STAGE_START.finditer(logs)
        if match.group(1) == model and int(match.group(2)) == batch
    ]
    last_start_position = start_matches[-1].start() if start_matches else -1
    if start_matches:
        _, _, phase, epoch, wd, source, output = start_matches[-1].groups()
        record["activePhase"] = phase
        record["activeEpoch"] = int(epoch)
        record["activeWd"] = wd
        record["activeWds"] = [wd]
        record["activeSourceCheckpoint"] = source
        record["activeOutput"] = output
    elif state in {"submitted", "scheduled"}:
        if record.get("policy") == FINALIZER_POLICY:
            requested = [int(epoch) for epoch in record.get("requestedPostDecayEpochs", [])]
            record["activePhase"] = "requested_post_decay_finalization"
            record["activeEpoch"] = requested[0] if requested else None
        else:
            record["activePhase"] = "backfill_pre_decay_evaluations"
            record["activeEpoch"] = int(record.get("historicalPreDecayStartEpoch", 8))

    saturation_matches = [
        match
        for match in SATURATED.finditer(logs)
        if match.group(1) == model and int(match.group(2)) == batch
    ]
    if saturation_matches:
        provisional_epoch = int(saturation_matches[-1].group(3))
        record["provisionalPreDecaySaturationEpoch"] = provisional_epoch
        if last_start_position < saturation_matches[-1].start():
            record["activePhase"] = "post_decay_selection"

    continuation_matches = [
        match
        for match in CONTINUED.finditer(logs)
        if match.group(1) == model and int(match.group(2)) == batch
    ]
    if continuation_matches:
        latest_continuation = continuation_matches[-1]
        decision = json.loads(latest_continuation.group(6))
        provisional_epoch = str(int(latest_continuation.group(3)))
        record.setdefault("postDecayContinuations", {})[provisional_epoch] = decision
        record["lastPostDecayContinuation"] = decision
        record.pop("postDecaySelection", None)
        record.pop("selectedPostDecayEpoch", None)
        record.pop("selectedPostDecayValidationExact", None)
        record.pop("selectedCheckpoint", None)
        record.pop("saturatedEpoch", None)
        record.pop("stopReason", None)
        record.pop("needsPolicyResume", None)
        if last_start_position < latest_continuation.start():
            record["activePhase"] = "continuing_constant_lr"
            record["activeEpoch"] = int(latest_continuation.group(5))

    selection_matches = [
        match
        for match in SELECTED.finditer(logs)
        if match.group(1) == model and int(match.group(2)) == batch
    ]
    if selection_matches:
        selection_match = selection_matches[-1]
        selection = json.loads(selection_match.group(7))
        postdecay_is_saturated = selection_postdecay_saturated(selection)
        declared = selection.get("postDecaySaturated")
        if declared is not None and bool(declared) != postdecay_is_saturated:
            raise RuntimeError(f"{model} BS{batch}: inconsistent post-decay decision")
        if not postdecay_is_saturated:
            record["supersededPostDecaySelection"] = selection
            record["needsPolicyResume"] = True
            record["status"] = "resume_required" if state == "complete" else state
            record["activePhase"] = "policy_resume_required"
            record["activeEpoch"] = int(selection["saturationEpoch"])
            record["activeWds"] = []
            record.pop("postDecaySelection", None)
            record.pop("selectedPostDecayEpoch", None)
            record.pop("selectedPostDecayValidationExact", None)
            record.pop("selectedCheckpoint", None)
            record.pop("saturatedEpoch", None)
            record.pop("stopReason", None)
        else:
            selection["postDecaySaturated"] = True
            selection["postDecaySaturationCriterion"] = (
                POST_DECAY_SATURATION_CRITERION
            )
            record.pop("needsPolicyResume", None)
            record.pop("supersededPostDecaySelection", None)
            record.update(
                {
                    "status": "complete",
                    "postDecaySelection": selection,
                    "selectedPostDecayEpoch": selection["selectedPostDecayEpoch"],
                    "selectedPostDecayValidationExact": selection[
                        "selectedPostDecayValidationExact"
                    ],
                    "selectedCheckpoint": selection["selectedCheckpoint"],
                    "saturatedEpoch": selection["saturationEpoch"],
                    "activePhase": "done",
                    "activeEpoch": None,
                    "activeWds": [],
                    "stopReason": (
                        f"Pre-decay validation provisionally saturated at "
                        f"E{selection['saturationEpoch']}; the newest of the three "
                        "post-decay results also failed to improve, and the selected "
                        "checkpoint is the best within that post-decay group."
                    ),
                }
            )

    finalized_matches = [
        match
        for match in FINALIZED.finditer(logs)
        if match.group(1) == model and int(match.group(2)) == batch
    ]
    if finalized_matches:
        finalization = json.loads(finalized_matches[-1].group(4))
        preserve = finalization.get("preserveExistingSelection") is True
        record.update(
            {
                "status": "complete",
                "beakerStatus": state,
                "activePhase": "done",
                "activeEpoch": None,
                "activeWds": [],
                "requestedPostDecayFinalizationResult": finalization,
                "postDecaySaturated": finalization.get("postDecaySaturated"),
                "stopReason": (
                    "The requested POST sequence reached strict non-improvement."
                    if finalization.get("postDecaySaturated") is True
                    else "Stopped at the exact user-requested frontier after completing the requested POST evaluations."
                ),
            }
        )
        transition = dict(record.get("requestedPostDecayFinalization", {}))
        transition.update(
            {
                "status": "complete",
                "completedPostDecayEpochs": finalization.get(
                    "requestedPostDecayEpochs", []
                ),
            }
        )
        record["requestedPostDecayFinalization"] = transition
        if not preserve:
            record.update(
                {
                    "postDecaySelection": finalization,
                    "selectedPostDecayEpoch": finalization[
                        "selectedPostDecayEpoch"
                    ],
                    "selectedPostDecayValidationExact": finalization[
                        "selectedPostDecayValidationExact"
                    ],
                    "selectedCheckpoint": finalization["selectedCheckpoint"],
                }
            )
            if finalization.get("postDecaySaturated") is True:
                record["saturatedEpoch"] = finalization["evaluatedThroughEpoch"]
            else:
                record.pop("saturatedEpoch", None)

    health = policy_health(logs, record, state)
    if health is not None:
        record["wandbHealth"] = health
        if health["shouldRecover"]:
            record["needsAttention"] = True
    elif record.get("policy") == FINALIZER_POLICY and state in {
        "submitted",
        "scheduled",
        "running",
    }:
        record["wandbHealth"] = {
            "status": "pending",
            "checkedAt": datetime.now(tz=UTC).isoformat(),
            "run": None,
            "url": None,
            "warnings": [],
            "criticalSignals": [],
            "shouldRecover": False,
            "reason": "The requested POST finalizer has not emitted a stage start yet.",
        }
        record.pop("needsAttention", None)
    if state == "failed" and not selection_matches:
        record["needsAttention"] = True

    step_values = legacy.TRAIN_STEP.findall(logs)
    train_values = legacy.TRAIN_LOSS.findall(logs)
    if step_values and record.get("activePhase") in {"constant_train", "post_decay_train"}:
        step, total = step_values[-1]
        step_value = int(step.replace(",", ""))
        total_value = int(total.replace(",", ""))
        record["progress"] = {
            "step": step_value,
            "totalSteps": total_value,
            "percent": round(100 * step_value / total_value, 1),
        }
        if train_values:
            record["progress"]["latestTrain"] = float(train_values[-1])
    elif record.get("activePhase") not in {"constant_train", "post_decay_train"}:
        record.pop("progress", None)
    pre_metrics = ", ".join(
        f"[PD] E{epoch} CE{result.get('validationExact', result.get('validation'))}"
        for epoch, result in sorted(
            record.get("preDecayResults", {}).items(), key=lambda item: int(item[0])
        )
    )
    post_metrics = ", ".join(
        f"[POST] E{epoch} CE{result.get('validationExact', result.get('validation'))}"
        for epoch, result in sorted(
            record.get("postDecayResults", {}).items(), key=lambda item: int(item[0])
        )
    )
    return (
        f"{model} BS{batch} {record['status']}; locked WD{record['lockedWd']}; "
        f"{pre_metrics or '[PD] awaiting evaluation'}; "
        f"{post_metrics or '[POST] not started'}; "
        f"phase={record.get('activePhase', 'pending')} E{record.get('activeEpoch') or 'done'}; "
        f"{experiment_id}"
    )


def main() -> None:
    summaries: list[str] = []
    for model, path in REPORTS.items():
        report = json.loads(path.read_text())
        for record in report.get("adaptiveDrWtEmbedWdChains", []):
            if record.get("policy") in POLICIES:
                summaries.append(update_policy_record(model, record))
        extend_report_display_epochs(report)
        report["updated"] = datetime.now(tz=UTC).date().isoformat()
        path.write_text(json.dumps(report, indent=2) + "\n")
        path.with_suffix(".js").write_text(
            "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
        )
    print("\n".join(summaries) if summaries else "no locked-WD chains registered yet")


if __name__ == "__main__":
    main()
