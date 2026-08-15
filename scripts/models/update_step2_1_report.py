#!/usr/bin/env python3
"""Refresh Step 2-1 unique-chain status and completed endpoint metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPORT_JSON = Path("reports/0802/data/wsd_unique_vs_repeated_batch_tuned_1b.json")
REPORT_JS = Path("reports/0802/data/wsd_unique_vs_repeated_batch_tuned_1b.js")
STEP1_JSON = Path("reports/0802/data/wsd_batch_size_1b.json")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUN_PATH = re.compile(r"/models/(?P<name>dense_1b_step2_1_0802_unique_[^/\s]+)/wandb/wandb/run-[^-\s]+-(?P<wandb>[a-z0-9]+)")
RUN_LINK = re.compile(r"View run (?P<name>\S+) at: https://wandb\.ai/[^\s]+/runs/(?P<wandb>[a-z0-9]+)")
LEGACY_RUN_NAME = re.compile(r"^dense_1b_step2_1_0802_unique_dclm5b_wsd_bs(?P<batch>64|256)_e(?P<epoch>[1-5])_lr(?P<lr>5e-4|1e-3|2e-3)_wd0\.033_warmup(?P<warmup>384|96)$")
FRACTIONAL_RUN_NAME = re.compile(r"^dense_1b_step2_1_0802_unique_dclm5b_wsd_bs(?P<batch>64|256)_e(?P<epoch>0p125|0p25|0p5)_lr(?P<lr>1e-3)_wd0\.033_warmup(?P<warmup>384|96)$")
EXTENSION_RUN_NAME = re.compile(r"^dense_1b_step2_1_0802_unique_ext1019b_wsd_bs(?P<batch>64|256|1024)_t(?P<epoch>6|8|10|12|16)b_lr(?P<lr>1e-3)_wd0\.033_warmup(?P<warmup>384|96|24)$")
CHAIN_START = re.compile(r"CHAIN_START bs=(?P<batch>64|256) lr=(?P<lr>\S+) target=(?P<epoch>[1-5])")
CHAIN_FINISH = re.compile(r"CHAIN_FINISH bs=(?P<batch>64|256) lr=(?P<lr>\S+) target=(?P<epoch>[1-5])")
FRACTIONAL_CHAIN_START = re.compile(r"FRACTIONAL_CHAIN_START bs=(?P<batch>64|256) lr=(?P<lr>1e-3) target=(?P<epoch>0\.125|0\.25|0\.5)")
FRACTIONAL_CHAIN_FINISH = re.compile(r"FRACTIONAL_CHAIN_FINISH bs=(?P<batch>64|256) lr=(?P<lr>1e-3) target=(?P<epoch>0\.125|0\.25|0\.5)")
EXTENSION_CHAIN_START = re.compile(r"EXTENSION_CHAIN_START bs=(?P<batch>64|256|1024) target=(?P<epoch>6|8|10|12|16)b")
EXTENSION_CHAIN_FINISH = re.compile(r"EXTENSION_CHAIN_FINISH bs=(?P<batch>64|256|1024) target=(?P<epoch>6|8|10|12|16)b")
TRAIN = re.compile(r"train/CE loss=([0-9.Ee+\-]+)")
HELDOUT = re.compile(r"dclm-validation-0802/CE loss=([0-9.Ee+\-]+)")
ACCURACY = re.compile(r"\s([a-z0-9_]+) \((?:length-normalized )?accuracy\)=([0-9.]+)")
BPB = re.compile(r"\s([a-z0-9_]+) \(BPB\)=([0-9.]+)")
AVERAGE_TASKS = ("arc_challenge", "arc_easy", "csqa", "hellaswag", "openbookqa", "piqa", "socialiqa", "winogrande")
TOKENS_PER_TARGET = 1_000_000_000
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
EXTENSION_TARGETS = (6, 8, 10, 12, 16)
FRACTIONAL_TARGETS = (0.125, 0.25, 0.5)
E5_PREDECAY = {
    64: "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_1_0802_unique_dclm5b_wsd_bs64_e5_lr1e-3_wd0.033_warmup384/step17166",
    256: "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_1_0802_unique_dclm5b_wsd_bs256_e5_lr1e-3_wd0.033_warmup96/step4291",
    1024: "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e5_lr1e-3_wd0.033_warmup24/step1073",
}


def match_run_name(name: str) -> tuple[re.Match[str], bool] | None:
    if match := LEGACY_RUN_NAME.match(name):
        return match, False
    if match := FRACTIONAL_RUN_NAME.match(name):
        return match, False
    if match := EXTENSION_RUN_NAME.match(name):
        return match, True
    return None


def target_value(raw: str) -> int | float:
    normalized = raw.replace("p", ".")
    value = float(normalized)
    return int(value) if value.is_integer() else value


def target_label(raw: str) -> str:
    return str(target_value(raw))


def stable_step(target: int | float, batch_sequences: int) -> int:
    end_step = math.ceil(target * TOKENS_PER_TARGET / (batch_sequences * SEQUENCE_LENGTH))
    return end_step - round(DECAY_FRACTION * end_step) - 1


def extension_output(batch_sequences: int, target: int, warmup: int) -> str:
    return (
        "/weka/oe-training-default/sewonm/icsl/models/"
        "dense_1b_step2_1_0802_unique_ext1019b_wsd_"
        f"bs{batch_sequences}_t{target}b_lr1e-3_wd0.033_warmup{warmup}"
    )


def normalize_task(name: str) -> str:
    for prefix, normalized in (("csqa", "csqa"), ("openbookqa", "openbookqa"), ("socialiqa", "socialiqa")):
        if name.startswith(prefix):
            return normalized
    return name


def command(*arguments: str) -> str:
    result = subprocess.run(arguments, check=True, stdout=subprocess.PIPE, text=True, errors="replace")
    return result.stdout


def experiment_state(experiment: str) -> tuple[str, str, dict[str, Any]]:
    payload = json.loads(command("beaker", "experiment", "get", experiment, "--format", "json"))
    record = payload[0] if isinstance(payload, list) else payload
    job = record["jobs"][0]
    timestamps = job.get("status", {})
    if "canceled" in timestamps:
        status = "canceled"
    elif "finalized" in timestamps or "exited" in timestamps:
        status = "complete" if timestamps.get("exitCode") == 0 else "failed"
    elif "started" in timestamps or "running" in timestamps or "resuming" in timestamps:
        status = "running"
    elif "scheduled" in timestamps or "created" in timestamps:
        status = "queued"
    else:
        status = "unknown"
    return job["id"], status, job


def parse_log(text: str) -> tuple[dict[str, dict[str, Any]], int | float | None, int | float]:
    runs: dict[str, dict[str, Any]] = {}
    current: str | None = None
    active_epoch: int | float | None = None
    finished_epoch: int | float = 0
    for raw in ANSI.sub("", text).splitlines():
        if match := EXTENSION_CHAIN_START.search(raw):
            active_epoch = int(match.group("epoch"))
        elif match := FRACTIONAL_CHAIN_START.search(raw):
            active_epoch = target_value(match.group("epoch"))
        elif match := CHAIN_START.search(raw):
            active_epoch = int(match.group("epoch"))
        if match := EXTENSION_CHAIN_FINISH.search(raw):
            finished_epoch = max(finished_epoch, int(match.group("epoch")))
        elif match := FRACTIONAL_CHAIN_FINISH.search(raw):
            finished_epoch = max(finished_epoch, target_value(match.group("epoch")))
        elif match := CHAIN_FINISH.search(raw):
            finished_epoch = max(finished_epoch, int(match.group("epoch")))
        if match := RUN_PATH.search(raw):
            name = match.group("name")
            if match_run_name(name) is not None:
                current = name
                runs.setdefault(current, {"wandb": match.group("wandb"), "downstream": {}, "downstreamBpb": {}})
            continue
        if match := RUN_LINK.search(raw):
            if match_run_name(match.group("name")) is not None:
                current = match.group("name")
                runs.setdefault(current, {"downstream": {}, "downstreamBpb": {}})["wandb"] = match.group("wandb")
            continue
        if current is None:
            continue
        run = runs[current]
        if match := TRAIN.search(raw):
            run["train"] = float(match.group(1))
        elif match := HELDOUT.search(raw):
            run["validation"] = float(match.group(1))
        elif match := ACCURACY.search(raw):
            run["downstream"][normalize_task(match.group(1))] = 100 * float(match.group(2))
        elif match := BPB.search(raw):
            run["downstreamBpb"][normalize_task(match.group(1))] = float(match.group(2))
    return runs, active_epoch, finished_epoch


def completed_rows(chain: dict[str, Any], parsed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in parsed.items():
        parsed_name = match_run_name(name)
        if parsed_name is None or "validation" not in metrics:
            continue
        match, is_extension = parsed_name
        downstream = metrics["downstream"]
        downstream_bpb = metrics["downstreamBpb"]
        if any(task not in downstream or task not in downstream_bpb for task in AVERAGE_TASKS):
            continue
        batch_sequences = int(match.group("batch"))
        target = target_value(match.group("epoch"))
        warmup = int(match.group("warmup"))
        output = (
            extension_output(batch_sequences, target, warmup)
            if is_extension
            else f"/weka/oe-training-default/sewonm/icsl/models/{name}"
        )
        row = {
            "batchSequences": int(match.group("batch")),
            "epoch": target,
            "lr": match.group("lr"),
            "wd": "0.033",
            "warmupSteps": int(match.group("warmup")),
            "status": "complete",
            "beaker": chain["beaker"],
            "job": chain["job"],
            "wandb": metrics["wandb"],
            "revision": chain["revision"],
            "output": output,
            "train": metrics.get("train"),
            "validation": metrics["validation"],
            "acc": downstream["hellaswag"],
            "bpb": downstream_bpb["hellaswag"],
            "avg8Bpb": sum(downstream_bpb[task] for task in AVERAGE_TASKS)
            / len(AVERAGE_TASKS),
            "downstream": downstream,
            "downstreamBpb": downstream_bpb,
            "reason": (
                "Completed inside the document-disjoint unique-extension chain with held-out DCLM validation and downstream evaluations."
                if is_extension
                else (
                    "Completed inside the persistent fractional unique-data Step 2-1 chain with held-out DCLM validation and downstream evaluations."
                    if target < 1
                    else "Completed inside the persistent 1B-through-5B Step 2-1 chain with held-out DCLM validation and downstream evaluations."
                )
            ),
        }
        if is_extension:
            previous_target = EXTENSION_TARGETS[EXTENSION_TARGETS.index(target) - 1] if target != EXTENSION_TARGETS[0] else None
            row["resumeCheckpoint"] = (
                E5_PREDECAY[batch_sequences]
                if previous_target is None
                else f"{extension_output(batch_sequences, previous_target, warmup)}/step{stable_step(previous_target, batch_sequences)}"
            )
            row["stableCheckpoint"] = f"{output}/step{stable_step(target, batch_sequences)}"
            row["extensionPoolPositionAtStableCheckpointTokens"] = (
                stable_step(target, batch_sequences)
                - int(E5_PREDECAY[batch_sequences].rsplit("step", 1)[1])
            ) * batch_sequences * SEQUENCE_LENGTH
        elif target < 1:
            row["resumeCheckpoint"] = (
                None
                if target == FRACTIONAL_TARGETS[0]
                else (
                    "/weka/oe-training-default/sewonm/icsl/models/"
                    f"dense_1b_step2_1_0802_unique_dclm5b_wsd_bs{batch_sequences}_"
                    f"e{str(FRACTIONAL_TARGETS[FRACTIONAL_TARGETS.index(target) - 1]).replace('.', 'p')}_"
                    f"lr{match.group('lr')}_wd0.033_warmup{warmup}/"
                    f"step{stable_step(FRACTIONAL_TARGETS[FRACTIONAL_TARGETS.index(target) - 1], batch_sequences)}"
                )
            )
            row["stableCheckpoint"] = f"{output}/step{stable_step(target, batch_sequences)}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = json.loads(REPORT_JSON.read_text())
    existing = {(r["batchSequences"], r["epoch"], r["lr"]): r for r in report.get("uniqueRuns", [])}
    for chain in report.get("chainExperiments", []):
        job, status, _ = experiment_state(chain["beaker"])
        chain["job"] = job
        log = command("beaker", "job", "logs", job, "--no-timestamps")
        parsed, active_epoch, finished_epoch = parse_log(log)
        chain["status"] = status
        targets = chain.get("targets", [1])
        if active_epoch is not None:
            chain["activeEpoch"] = active_epoch
        else:
            chain["activeEpoch"] = next(
                (target for target in targets if target > finished_epoch), targets[-1]
            )
        phase_wandb: dict[str, str] = {}
        for name, run in parsed.items():
            parsed_name = match_run_name(name)
            if parsed_name and run.get("wandb"):
                phase_wandb[target_label(parsed_name[0].group("epoch"))] = run["wandb"]
        chain["phaseWandb"] = phase_wandb
        for row in completed_rows(chain, parsed):
            existing[(row["batchSequences"], row["epoch"], row["lr"])] = row
    report["uniqueRuns"] = sorted(existing.values(), key=lambda r: (r["batchSequences"], r["epoch"], float(r["lr"])))
    step1 = json.loads(STEP1_JSON.read_text())
    # The Step 1-1 `updated` field also carries a free-form activity note.  Keep
    # only its timestamp here so stale/independent sweep policy text is not
    # presented as part of the Step 2-1 comparison.
    repeated_updated = step1.get("updated")
    report["repeatedSourceUpdated"] = (
        repeated_updated.split(" · ", 1)[0] if repeated_updated else None
    )
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PDT")
    complete = len(report["uniqueRuns"])
    active = sum(chain["status"] in {"queued", "running"} for chain in report["chainExperiments"])
    report["updated"] = f"{now} · {complete} unique endpoints complete; {active} chains active/queued; repeated results refreshed from Step 1-1"
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.dry_run:
        print(rendered)
        return
    REPORT_JSON.write_text(rendered)
    REPORT_JS.write_text("window.STEP2_1_REPORT_DATA = " + rendered.rstrip() + ";\n")


if __name__ == "__main__":
    main()
