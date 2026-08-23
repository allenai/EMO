#!/usr/bin/env python3
"""Refresh Dense-1B all-POST coordinates and enforce cross-WD pruning."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/0802/data/wsd_data_loader_1b.json")
REPORT_JS_PATH = REPORT_PATH.with_suffix(".js")
POLICY = "dense_1b_all_postdecay_saturation_v1"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RESULT = re.compile(
    r"DENSE1B_PDPOST_RESULT bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"phase=(pre_decay|post_decay) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
STAGE_START = re.compile(
    r"DENSE1B_PDPOST_STAGE_START bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"phase=([^ ]+) epoch=([0-9]+)(?: previous_epoch=([0-9]+))? "
    r"source=([^ ]+) output=([^\s]+)"
)
TERMINAL = re.compile(
    r"DENSE1B_PDPOST_(SELECTED|PRUNED) bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"trigger_epoch=([0-9]+) selected_epoch=([0-9]+) validation=([^ ]+) json=(\{.*\})$",
    re.MULTILINE,
)
CONTINUE = re.compile(
    r"DENSE1B_PDPOST_CONTINUE bs=([0-9]+) lr=([^ ]+) wd=([^ ]+) "
    r"trigger_epoch=([0-9]+).*?json=(\{.*\})$",
    re.MULTILINE,
)
TRAIN_STEP = re.compile(r"\[step=([0-9,]+)/([0-9,]+),epoch=")
TRAIN_LOSS = re.compile(r"\btrain/CE loss=([^\s]+)")
WANDB_RUN = re.compile(r"https://wandb\.ai/[^\s]+/runs/([a-zA-Z0-9_-]+)")
TERMINAL_STATES = {"complete", "pruned", "not_required"}


def run(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return completed.stdout


def write_report(report: dict[str, Any]) -> None:
    report["updated"] = datetime.now(tz=UTC).date().isoformat()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS_PATH.write_text(
        "window.ICSL_DATA_LOADER_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def inspect_experiment(experiment: str) -> dict[str, Any]:
    payload = json.loads(run(["beaker", "experiment", "inspect", experiment, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one experiment for {experiment}")
    return payload[0]


def experiment_state(experiment: dict[str, Any]) -> str:
    jobs = experiment.get("jobs") or []
    if not jobs:
        return "submitted"
    statuses = [job.get("status") or {} for job in jobs]
    terminal = {"exited", "finalized", "canceled", "cancelled"}
    if any("started" in status and not terminal.intersection(status) for status in statuses):
        return "running"
    if any(
        "scheduled" in status and "started" not in status and not terminal.intersection(status)
        for status in statuses
    ):
        return "scheduled"
    if any("finalized" in status and status.get("exitCode") == 0 for status in statuses):
        return "complete"
    if all(terminal.intersection(status) for status in statuses):
        return "failed"
    return "submitted"


def coordinate_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in report.get("runs", []) if record.get("policy") == POLICY]


def parse_health(record: dict[str, Any], logs: str, state: str) -> dict[str, Any] | None:
    starts = list(STAGE_START.finditer(logs))
    if not starts:
        return None
    _batch, lr, wd, phase, epoch, previous, source, output = starts[-1].groups()
    segment = logs[starts[-1].start() :]
    steps = [
        (int(step.replace(",", "")), int(total.replace(",", "")))
        for step, total in TRAIN_STEP.findall(segment)
    ]
    nonfinite: list[str] = []
    finite: list[float] = []
    for raw in TRAIN_LOSS.findall(segment):
        try:
            value = float(raw)
        except ValueError:
            nonfinite.append(raw)
            continue
        if math.isfinite(value):
            finite.append(value)
        else:
            nonfinite.append(raw)
    critical: list[str] = []
    if nonfinite:
        critical.append("nonfinite-training-loss")
    expected = str(record["output"])
    if (
        "eval" not in phase
        and output != expected
        and not output.startswith(expected + "/.all_postdecay_policy")
    ):
        critical.append("stage-output-path-mismatch")
    if previous and int(previous) > 0 and "Loading checkpoint from '" not in segment:
        critical.append("missing-resume-checkpoint-load")
    run_ids = WANDB_RUN.findall(segment)
    return {
        "status": "critical" if critical else "healthy" if steps else "pending",
        "checkedAt": datetime.now(tz=UTC).isoformat(),
        "run": run_ids[-1] if run_ids else None,
        "url": f"https://wandb.ai/ai2-llm/sewonm-icsl/runs/{run_ids[-1]}" if run_ids else None,
        "phase": phase,
        "epoch": int(epoch),
        "lr": lr,
        "wd": wd,
        "source": source,
        "output": output,
        "latestStep": steps[-1][0] if steps else None,
        "totalSteps": steps[-1][1] if steps else None,
        "latestTrain": finite[-1] if finite else None,
        "nonfiniteLosses": nonfinite[:5],
        "criticalSignals": critical,
        "shouldRecover": bool(critical),
        "beakerState": state,
    }


def refresh_coordinate(record: dict[str, Any]) -> str:
    experiment = record.get("experiment")
    if not experiment:
        return f"{record['id']}: {record.get('status', 'planned')}"
    inspected = inspect_experiment(str(experiment))
    state = experiment_state(inspected)
    jobs = [job for job in inspected.get("jobs") or [] if job.get("id")]
    record["beakerStatus"] = state
    if jobs:
        record["job"] = jobs[-1]["id"]
        record["jobs"] = [job["id"] for job in jobs]
    logs = ""
    if state in {"running", "complete", "failed"}:
        logs = ANSI.sub("", run(["beaker", "experiment", "logs", str(experiment)]))
    for batch, lr, wd, phase, epoch, payload in RESULT.findall(logs):
        if (
            int(batch) != int(record["batchSequences"])
            or lr != str(record["lr"])
            or wd != str(record["wd"])
        ):
            continue
        result = json.loads(payload)
        result.setdefault("status", "complete")
        result.update(
            {
                "epoch": int(epoch),
                "lr": lr,
                "wd": wd,
                "experiment": experiment,
                "job": record.get("job"),
                "revision": record.get("revision"),
            }
        )
        target = record.setdefault(
            "preDecayResults" if phase == "pre_decay" else "postDecayResults", {}
        )
        target[str(int(epoch))] = result
        predecay = record.get("preDecayResults", {}).get(str(int(epoch)))
        postdecay = record.get("postDecayResults", {}).get(str(int(epoch)))
        combined: dict[str, Any] = {
            "status": (postdecay or predecay or {}).get("status", "complete"),
            "preDecay": predecay,
            "postDecay": postdecay,
        }
        # Backward-compatible summary fields intentionally expose POST only.
        # This prevents legacy report selectors from comparing PD with POST.
        if postdecay:
            for key in ("validation", "validationExact", "train", "gap", "wandb"):
                if key in postdecay:
                    combined[key] = postdecay[key]
        record.setdefault("results", {})[str(int(epoch))] = combined
    starts = [
        match.groups()
        for match in STAGE_START.finditer(logs)
        if int(match.group(1)) == int(record["batchSequences"])
        and match.group(2) == str(record["lr"])
        and match.group(3) == str(record["wd"])
    ]
    if starts:
        _, _, _, phase, epoch, _, source, output = starts[-1]
        record.update(
            {
                "activePhase": phase,
                "activeEpoch": int(epoch),
                "activeSource": source,
                "activeOutput": output,
            }
        )
    for kind, batch, lr, wd, trigger, selected, validation, payload in TERMINAL.findall(logs):
        if (
            int(batch) != int(record["batchSequences"])
            or lr != str(record["lr"])
            or wd != str(record["wd"])
        ):
            continue
        selection = json.loads(payload)
        record.update(
            {
                "status": "pruned" if kind == "PRUNED" else "complete",
                "activePhase": None,
                "activeEpoch": None,
                "postDecaySelection": selection,
                "selectedPostDecayEpoch": int(selected),
                "selectedPostDecayValidationExact": float(validation),
            }
        )
    continuations = [
        json.loads(payload)
        for batch, lr, wd, _, payload in CONTINUE.findall(logs)
        if int(batch) == int(record["batchSequences"])
        and lr == str(record["lr"])
        and wd == str(record["wd"])
    ]
    if continuations:
        record["latestPostDecayContinuation"] = continuations[-1]
    if record.get("status") not in {"complete", "pruned"}:
        record["status"] = state
    health = parse_health(record, logs, state)
    if health:
        record["wandbHealth"] = health
        if health["shouldRecover"]:
            record["needsAttention"] = True
    step_values = TRAIN_STEP.findall(logs)
    if step_values and record.get("activeEpoch") is not None:
        step, total = step_values[-1]
        step_value = int(step.replace(",", ""))
        total_value = int(total.replace(",", ""))
        record["progress"] = {
            "step": step_value,
            "totalSteps": total_value,
            "percent": round(100 * step_value / total_value, 1),
        }
    else:
        record.pop("progress", None)
    if state == "failed" and record.get("status") not in {"complete", "pruned"}:
        record["needsAttention"] = True
        record["reason"] = "Beaker exhausted retries before a terminal POST decision."
    return (
        f"{record['id']}: {record['status']} phase={record.get('activePhase') or 'done'} "
        f"E{record.get('activeEpoch') or '—'} PD={sorted(map(int, record.get('preDecayResults', {})))} "
        f"POST={sorted(map(int, record.get('postDecayResults', {})))} {experiment}"
    )


def wd_prune_requests(report: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    active_statuses = {"submitted", "scheduled", "running"}
    records = coordinate_records(report)
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("variant") != "DR+WT+EmbedWD":
            continue
        groups.setdefault(
            (int(record["batchSequences"]), str(record["lr"]), str(record["variant"])), []
        ).append(record)
    for (batch, lr, _), group in groups.items():
        if len(group) != 2:
            continue
        lower, higher = sorted(group, key=lambda item: Decimal(str(item["wd"])))
        if Decimal(str(higher["wd"])) != Decimal("1.0"):
            continue
        low_results = lower.get("postDecayResults", {})
        high_results = higher.get("postDecayResults", {})
        common_epochs = sorted(set(low_results) & set(high_results), key=int)
        winning = [
            epoch
            for epoch in common_epochs
            if Decimal(str(high_results[epoch]["validationExact"]))
            < Decimal(str(low_results[epoch]["validationExact"]))
        ]
        if not winning:
            continue
        epoch = int(winning[0])
        evidence = {
            "batchSequences": batch,
            "lr": lr,
            "epoch": epoch,
            "lowerWd": str(lower["wd"]),
            "higherWd": str(higher["wd"]),
            "lowerValidationExact": float(low_results[str(epoch)]["validationExact"]),
            "higherValidationExact": float(high_results[str(epoch)]["validationExact"]),
            "comparisonGroup": "post_decay",
            "criterion": "strict_higher_wd_post_win_same_bs_lr_epoch",
        }
        lower["higherWdWinEvidence"] = evidence
        if len(low_results) < 3:
            lower["pruneDeferredReason"] = "fewer_than_three_completed_post_sources"
            continue
        if lower.get("status") not in active_statuses:
            continue
        if str(lower.get("activePhase", "")).startswith("post_decay"):
            lower["pruneDeferredReason"] = "already_finalizing_post_decay"
            continue
        lower["pruneRequested"] = evidence
        lower.pop("pruneDeferredReason", None)
        requests.append(lower)
    return requests


def aggregate_chain(report: dict[str, Any], chain: dict[str, Any]) -> None:
    if chain["id"] == "dense-1b-bs512-lr2e-3-conditional-followup":
        identifiers = {
            record["id"]
            for record in coordinate_records(report)
            if int(record["batchSequences"]) == 512
        }
    else:
        identifiers = {
            record["id"]
            for record in coordinate_records(report)
            if int(record["batchSequences"]) == int(chain["batchSequences"])
            and record.get("method") != "conditional512"
        }
    records = [record for record in coordinate_records(report) if record["id"] in identifiers]
    if not records:
        return
    chain["coordinateStates"] = {record["id"]: record.get("status") for record in records}
    if all(record.get("status") in TERMINAL_STATES for record in records):
        invalid_complete = [
            record["id"]
            for record in records
            if record.get("status") == "complete"
            and (record.get("postDecaySelection") or {}).get("postDecaySaturated") is not True
        ]
        if invalid_complete:
            chain["status"] = "failed"
            chain["reason"] = (
                "Coordinates marked complete without confirmed POST saturation: "
                + ", ".join(invalid_complete)
            )
            return
        selections = [
            record
            for record in records
            if record.get("postDecaySelection")
            and record.get("selectedPostDecayValidationExact") is not None
        ]
        if not selections and all(record.get("status") == "not_required" for record in records):
            chain["status"] = "not_required"
            return
        if len(selections) != len([r for r in records if r.get("status") != "not_required"]):
            chain["status"] = "failed"
            chain["reason"] = "A terminal coordinate lacks its required POST selection."
            return
        selected = min(
            selections,
            key=lambda record: (
                Decimal(str(record["selectedPostDecayValidationExact"])),
                Decimal(str(record["lr"])),
                -Decimal(str(record["wd"])),
            ),
        )
        chain.update(
            {
                "status": "complete",
                "selectedRun": selected["id"],
                "selectedLr": str(selected["lr"]),
                "selectedWd": str(selected["wd"]),
                "selectedVariant": str(selected["variant"]),
                "selectedPostDecayEpoch": selected["selectedPostDecayEpoch"],
                "selectedPostDecayValidationExact": selected["selectedPostDecayValidationExact"],
                "reason": "All coordinate jobs are terminal; winner selected only from POST results.",
            }
        )
    elif any(record.get("status") == "failed" for record in records):
        chain["status"] = "failed"
    elif any(record.get("status") == "running" for record in records):
        chain["status"] = "running"
    elif any(record.get("status") == "scheduled" for record in records):
        chain["status"] = "scheduled"
    elif any(record.get("experiment") for record in records):
        chain["status"] = "submitted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune-if-ready", action="store_true")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.prune_if_ready and not args.revision:
        parser.error("--revision is required with --prune-if-ready")
    report = json.loads(REPORT_PATH.read_text())
    records = coordinate_records(report)
    summaries = [refresh_coordinate(record) for record in records]
    prune = wd_prune_requests(report)
    for chain in report.get("drWtEmbedWdGridChains", []):
        aggregate_chain(report, chain)
    write_report(report)
    if args.prune_if_ready:
        for record in prune:
            output = run(
                [
                    ".venv/bin/python",
                    "scripts/models/submit_dense_1b_dr_wt_embedwd_grid.py",
                    "--finalize-run",
                    record["id"],
                    "--revision",
                    args.revision,
                ]
            )
            summaries.append(output.strip())
    elif prune:
        summaries.extend(f"PRUNE READY {record['id']}" for record in prune)
    print("\n".join(summaries) if summaries else "no Dense-1B all-POST coordinates registered")


if __name__ == "__main__":
    main()
