#!/usr/bin/env python3
"""Generate an apply_patch payload for completed persistent WSD chains."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from parse_wsd_chain_logs import parse_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-js", type=Path, required=True)
    parser.add_argument("--model-key", choices=("small", "medium"), required=True)
    parser.add_argument("--updated", required=True)
    parser.add_argument("experiment", nargs="+")
    return parser.parse_args()


def property_bounds(text: str, key: str) -> tuple[int, int]:
    marker = f'  "{key}": '
    start = text.index(marker)
    value_start = start + len(marker)
    opening = text[value_start]
    closing = {"[": "]", "{": "}"}[opening]
    depth = 0
    in_string = False
    escaped = False
    end = value_start
    for index in range(value_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    return start, end


def append_property_entries(text: str, key: str, entries: list[dict[str, object]]) -> tuple[str, str, str]:
    if not entries:
        return text, "", ""
    marker = f'  "{key}": ['
    if marker not in text:
        raise ValueError(f"property marker not found: {key}")
    rendered_entries = ["    " + json.dumps(entry, ensure_ascii=False, separators=(", ", ": ")) for entry in entries]
    new = marker + "\n" + ",\n".join(rendered_entries) + ","
    return text.replace(marker, new, 1), marker, new


def patch_hunk(old: str, new: str) -> str:
    return "@@\n" + "\n".join(f"-{line}" for line in old.splitlines()) + "\n" + "\n".join(
        f"+{line}" for line in new.splitlines()
    )


def main() -> None:
    args = parse_args()
    data = json.loads(args.report_json.read_text())
    parsed: list[tuple[str, dict[str, object]]] = []
    for experiment in args.experiment:
        result = subprocess.run(
            ["beaker", "experiment", "logs", experiment, "--hide-source"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        for record in parse_lines(result.stdout.splitlines(), skip_incomplete=True):
            parsed.append((experiment, record))

    additions: dict[str, list[dict[str, object]]] = {
        f"{args.model_key}UniqueRuns": [],
        f"{args.model_key}RepeatedRuns": [],
    }
    for experiment, record in parsed:
        regime = str(record.pop("regime"))
        name = str(record.pop("name"))
        key = f"{args.model_key}{'Unique' if regime == 'unique' else 'Repeated'}Runs"
        entry = {
            "epoch": record.pop("epoch"),
            "lr": record.pop("lr"),
            "wd": "0.033",
            "status": "complete",
            "initSeed": 12536,
            "dataSeed": 0,
            "beaker": experiment,
            **record,
            "output": f"/weka/oe-training-default/sewonm/icsl/models/{name}",
            "reason": "Completed successfully inside the persistent LR chain with held-out DCLM validation and all nine downstream evaluations.",
        }
        existing = data[key]
        identity = (entry["epoch"], entry["lr"], entry["wandb"])
        if not any((item.get("epoch"), item.get("lr"), item.get("wandb")) == identity for item in existing):
            existing.append(entry)
            additions[key].append(entry)
    data["updated"] = args.updated

    json_text = args.report_json.read_text()
    js_text = args.report_js.read_text()
    changes: dict[Path, list[tuple[str, str]]] = {args.report_json: [], args.report_js: []}
    for key in (f"{args.model_key}UniqueRuns", f"{args.model_key}RepeatedRuns"):
        json_text, old, new = append_property_entries(json_text, key, additions[key])
        if old:
            changes[args.report_json].append((old, new))
        js_text, old, new = append_property_entries(js_text, key, additions[key])
        if old:
            changes[args.report_js].append((old, new))
    print("*** Begin Patch")
    for path, hunks in changes.items():
        if not hunks:
            continue
        print(f"*** Update File: {path.resolve()}")
        for old, new in hunks:
            print(patch_hunk(old, new))
    print("*** End Patch")


if __name__ == "__main__":
    main()
