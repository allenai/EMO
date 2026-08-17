#!/usr/bin/env python3
"""Discover and evaluate Dense-1B DR checkpoints serially on one GPU."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from evaluate_dense_1b_missing_downstream import evaluation_arguments, replace_argument, task_name

MODELS_ROOT = Path("/weka/oe-training-default/sewonm/icsl/models")
STEP = re.compile(r"step(\d+)$")


def variant_matches(name: str, variant: str) -> bool:
    if variant == "dr-wt-embwd":
        return "_dr_wt_embwd_" in name
    if variant == "dr-wt":
        return "_dr_wt_" in name and "_embwd_" not in name
    return "_dr_" in name and "_wt_" not in name


def expected_output_name(item: dict) -> str:
    variant = {
        "dr": "dr",
        "dr-wt": "dr_wt",
        "dr-wt-embwd": "dr_wt_embwd",
    }[item["variant"]]
    return (
        f"bs{item['batchSequences']}_{variant}_"
        f"lr{item['lr']}_wd{item['wd']}"
    )


def discover_output_directories() -> list[Path]:
    directories: list[Path] = []
    for path in MODELS_ROOT.rglob("*_dr_*"):
        if not path.is_dir() or path.name.startswith("step"):
            continue
        if any(child.is_dir() and STEP.fullmatch(child.name) for child in path.iterdir()):
            directories.append(path)
    return sorted(set(directories))


def resolve_checkpoint(item: dict, outputs: list[Path]) -> Path | None:
    registered = Path(item["checkpoint"])
    if registered.is_dir():
        return registered
    step = registered.name
    batch = f"bs{item['batchSequences']}_"
    lr = f"lr{item['lr']}"
    wd = f"wd{item['wd']}"
    matches = [
        output / step
        for output in outputs
        if batch in output.name
        and lr in output.name
        and wd in output.name
        and variant_matches(output.name, item["variant"])
        and (output / step).is_dir()
    ]
    exact_name = [path for path in matches if path.parent.name == expected_output_name(item)]
    if exact_name:
        matches = exact_name
    if len(matches) > 1:
        canonical = [path for path in matches if path.parent.parent.name == "dense_1b_dclm1b"]
        if len(canonical) == 1:
            matches = canonical
    if len(matches) > 1:
        paths = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"ambiguous runtime checkpoint for {item['runId']} E{item['epoch']}:\n{paths}"
        )
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-id")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    base_arguments = [
        argument
        for argument in manifest["baseArguments"]
        if argument != "--decay-embeddings"
        and not argument.startswith("--mlp-weight-decay=")
        and not argument.startswith("--mlp-weight-decay-scope=")
    ]
    items = manifest["items"]
    if args.group_id:
        items = [item for item in items if item["groupId"] == args.group_id]
        if not items:
            raise RuntimeError(f"manifest contains no downstream group {args.group_id}")
    outputs = discover_output_directories()
    print(
        f"DOWNSTREAM_DISCOVERY_COMPLETE root={MODELS_ROOT} dr_output_directories={len(outputs)}",
        flush=True,
    )
    for item in items:
        checkpoint = resolve_checkpoint(item, outputs)
        name = task_name(item)
        arguments = replace_argument(
            base_arguments,
            "--model.tie_embeddings=",
            f"--model.tie_embeddings={'false' if item['variant'] == 'dr' else 'true'}",
        )
        resolved_item = dict(item)
        if checkpoint is not None:
            resolved_item["checkpoint"] = str(checkpoint)
        arguments = evaluation_arguments(arguments, resolved_item, name)
        marker = f"run_id={item['runId']} epoch={item['epoch']}"
        if checkpoint is None:
            print(
                f"DOWNSTREAM_EVAL_SKIPPED {marker} reason=missing_checkpoint ",
                f"registered_checkpoint={item['checkpoint']}",
                flush=True,
            )
            continue
        print(
            f"DOWNSTREAM_EVAL_START {marker} checkpoint={checkpoint} "
            f"registered_checkpoint={item['checkpoint']}",
            flush=True,
        )
        subprocess.run(
            ["python", manifest["trainingScript"], name, "--dry-run", *arguments], check=True
        )
        subprocess.run(
            [
                "torchrun",
                "--standalone",
                "--nproc-per-node=1",
                manifest["trainingScript"],
                name,
                *arguments,
            ],
            check=True,
        )
        print(f"DOWNSTREAM_EVAL_COMPLETE {marker}", flush=True)


if __name__ == "__main__":
    main()
