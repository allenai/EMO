#!/usr/bin/env python3
"""Evaluate a manifest of Dense-1B checkpoints serially on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from evaluate_dense_1b_missing_downstream import evaluation_arguments, replace_argument, task_name


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
    for item in items:
        name = task_name(item)
        arguments = replace_argument(
            base_arguments,
            "--model.tie_embeddings=",
            f"--model.tie_embeddings={'false' if item['variant'] == 'dr' else 'true'}",
        )
        arguments = evaluation_arguments(arguments, item, name)
        marker = f"run_id={item['runId']} epoch={item['epoch']}"
        print(f"DOWNSTREAM_EVAL_START {marker} checkpoint={item['checkpoint']}", flush=True)
        if not Path(item["checkpoint"]).is_dir():
            raise FileNotFoundError(item["checkpoint"])
        subprocess.run(
            ["python", manifest["trainingScript"], name, "--dry-run", *arguments], check=True
        )
        subprocess.run(
            [
                "torchrun",
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
