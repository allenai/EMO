#!/usr/bin/env python3
"""Extract final per-stage WSD metrics from a persistent Beaker chain log."""

from __future__ import annotations

import json
import re
import sys


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUN_PATH = re.compile(r"/models/([^/]+)/wandb/wandb/run-[^-\s]+-([a-z0-9]+)")
RUN_LINK = re.compile(r"View run (\S+) at: https://wandb\.ai/[^\s]+/runs/([a-z0-9]+)")
ACCURACY = re.compile(r"\s([a-z0-9_]+) \((?:length-normalized )?accuracy\)=([0-9.]+)")
BPB = re.compile(r"\s([a-z0-9_]+) \(BPB\)=([0-9.]+)")
HELDOUT = re.compile(r"dclm-validation-0802/CE loss=([0-9.]+)")
RUN_NAME = re.compile(
    r"^(?P<prefix>dense|stdmoe)_(?P<size>153m(?:720m)?|474m(?:3b5)?)_step2_0802_"
    r"(?P<regime>unique|repeated)_dclm(?:5b|1b)_wsd_e(?P<epoch>1|2|4|8|12|16)_"
    r"lr(?P<lr>1e-3|2e-3|4e-3)_wd0\.033_warmup(?:95|48)$"
)


def normalized_task(name: str) -> str:
    if name.startswith("csqa"):
        return "csqa"
    if name.startswith("openbookqa"):
        return "openbookqa"
    if name.startswith("socialiqa"):
        return "socialiqa"
    return name


def parse_lines(lines: object, *, skip_incomplete: bool = False) -> list[dict[str, object]]:
    runs: dict[str, dict[str, object]] = {}
    current: str | None = None
    for raw_line in lines:  # type: ignore[union-attr]
        line = ANSI.sub("", raw_line)
        if match := RUN_PATH.search(line):
            current, wandb_id = match.groups()
            if RUN_NAME.match(current):
                runs.setdefault(current, {"wandb": wandb_id, "downstream": {}, "downstreamBpb": {}})
            else:
                current = None
            continue
        if current is None:
            continue
        run = runs[current]
        if match := ACCURACY.search(line):
            task, value = match.groups()
            run["downstream"][normalized_task(task)] = 100 * float(value)  # type: ignore[index]
        elif match := BPB.search(line):
            task, value = match.groups()
            run["downstreamBpb"][normalized_task(task)] = float(value)  # type: ignore[index]
        elif match := HELDOUT.search(line):
            run["c4"] = float(match.group(1))
        elif match := RUN_LINK.search(line):
            name, wandb_id = match.groups()
            if name == current:
                run["wandb"] = wandb_id

    output: list[dict[str, object]] = []
    average_tasks = (
        "arc_challenge",
        "arc_easy",
        "csqa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    )
    for name, run in runs.items():
        match = RUN_NAME.match(name)
        assert match is not None
        downstream = run["downstream"]
        downstream_bpb = run["downstreamBpb"]
        missing = [task for task in average_tasks if task not in downstream or task not in downstream_bpb]  # type: ignore[operator]
        if "c4" not in run or missing:
            if skip_incomplete:
                continue
            raise RuntimeError(f"Incomplete metrics for {name}: c4={run.get('c4')}, missing={missing}")
        output.append(
            {
                "name": name,
                "regime": match.group("regime"),
                "epoch": int(match.group("epoch")),
                "lr": match.group("lr"),
                "wandb": run["wandb"],
                "c4": run["c4"],
                "acc": downstream["hellaswag"],  # type: ignore[index]
                "bpb": downstream_bpb["hellaswag"],  # type: ignore[index]
                "avg8Bpb": sum(downstream_bpb[task] for task in average_tasks) / len(average_tasks),  # type: ignore[index]
                "downstream": downstream,
                "downstreamBpb": downstream_bpb,
            }
        )
    return sorted(output, key=lambda run: (run["regime"], run["epoch"], run["lr"]))


def main() -> None:
    print(json.dumps(parse_lines(sys.stdin)))


if __name__ == "__main__":
    main()
