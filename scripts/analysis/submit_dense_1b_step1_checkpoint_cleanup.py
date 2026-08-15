#!/usr/bin/env python3
"""Inventory or delete surplus Dense-1B Step-1 checkpoints on Weka.

The remote task is deliberately narrow: it scans only direct children of the
1B Step-1 0802 model prefix, protects every checkpoint in explicitly active
outputs, and retains endpoint checkpoints for E1, E2, and every multiple of
four.  Delete mode requires the exact SHA-256 manifest produced by a previous
inventory run, so a changing candidate set fails closed.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import shlex
import subprocess
import tempfile
import textwrap
import zlib
from pathlib import Path
from typing import Any


DEFAULT_BASE_EXPERIMENT = "01KZKZ5DG8HFHBKKSC0V7AWW7S"
DEFAULT_ROOT = "/weka/oe-training-default/sewonm/icsl/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inventory", "delete"), required=True)
    parser.add_argument("--expected-hash")
    parser.add_argument("--base-experiment", default=DEFAULT_BASE_EXPERIMENT)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--name", required=True)
    parser.add_argument("--protect-output", action="append", default=[])
    parser.add_argument(
        "--exclude-batch",
        action="append",
        type=int,
        choices=(16, 32, 64, 128, 256, 512, 1024),
        default=[],
    )
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "delete" and not args.expected_hash:
        parser.error("--expected-hash is required in delete mode")
    if args.mode == "inventory" and args.expected_hash:
        parser.error("--expected-hash is only valid in delete mode")
    for output in args.protect_output:
        if not output.startswith(args.root + "/"):
            parser.error(f"protected output is outside cleanup root: {output}")
    return args


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def remote_program(config: dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode()
    return textwrap.dedent(
        f"""
        import base64
        import collections
        import hashlib
        import json
        import math
        import os
        import re
        import shutil
        import subprocess
        import zlib
        from pathlib import Path

        config = json.loads(base64.b64decode({encoded!r}))
        root = Path(config["root"]).resolve()
        required_prefix = str(root) + "/"
        active_outputs = {{str(Path(path)) for path in config["activeOutputs"]}}
        excluded_batches = set(config["excludedBatches"])
        allowed_exact = {{1, 2}}
        batch_pattern = re.compile(r"_bs(16|32|64|128|256|512|1024)(?:_|$)")
        warmup_pattern = re.compile(r"_warmup(1536|768|384|192|96|48|24)(?:_|$)")
        epoch_pattern = re.compile(r"_e(0p125|0p25|0p5|[0-9]+)(?:_|-|$)")
        checkpoint_pattern = re.compile(r"step([0-9]+)(-tmp)?$")
        warmup_batch = {{1536: 16, 768: 32, 384: 64, 192: 128, 96: 256, 48: 512, 24: 1024}}

        def allowed_epoch(value):
            return value in allowed_exact or (value >= 4 and value % 4 == 0)

        def infer_batch(name):
            match = batch_pattern.search(name)
            if match:
                return int(match.group(1))
            match = warmup_pattern.search(name)
            return warmup_batch.get(int(match.group(1))) if match else None

        def infer_target_epoch(name):
            match = epoch_pattern.search(name)
            if not match:
                return None
            text = match.group(1)
            if text == "0p125":
                return 0.125
            if text == "0p25":
                return 0.25
            if text == "0p5":
                return 0.5
            return int(text)

        parents = []
        candidates = []
        protected = []
        unparseable = []
        candidate_reasons = collections.Counter()
        protected_reasons = collections.Counter()
        for parent in sorted(root.iterdir()):
            if not parent.is_dir() or not parent.name.startswith("dense_1b_step1_0802_"):
                continue
            children = []
            for child in parent.iterdir():
                if not child.is_dir():
                    continue
                match = checkpoint_pattern.fullmatch(child.name)
                if match:
                    children.append((int(match.group(1)), bool(match.group(2)), child))
            if not children:
                continue
            parents.append(str(parent))
            if str(parent) in active_outputs:
                for _, _, path in children:
                    protected.append(str(path))
                    protected_reasons["active-output"] += 1
                continue
            batch = infer_batch(parent.name)
            target_epoch = infer_target_epoch(parent.name)
            if batch is None or target_epoch is None:
                unparseable.append(str(parent))
                for _, _, path in children:
                    protected.append(str(path))
                    protected_reasons["unparseable-output"] += 1
                continue
            if batch in excluded_batches:
                for _, _, path in children:
                    protected.append(str(path))
                    protected_reasons["excluded-batch"] += 1
                continue
            complete = sorted((step, path) for step, temporary, path in children if not temporary)
            keep = set()
            if allowed_epoch(target_epoch) and complete:
                keep.add(str(complete[-1][1]))
            steps_per_epoch = 1_000_000_000 / (batch * 4096)
            max_epoch = max(512, int(target_epoch) if target_epoch >= 1 else 1)
            expected = set()
            for epoch in range(1, max_epoch + 1):
                if not allowed_epoch(epoch):
                    continue
                post_decay = epoch * steps_per_epoch
                pre_decay = post_decay * 0.9
                for raw in (pre_decay, post_decay):
                    expected.update({{math.floor(raw), round(raw), math.ceil(raw)}})
            for step, temporary, path in children:
                path_text = str(path)
                if not temporary and any(abs(step - value) <= 1 for value in expected):
                    keep.add(path_text)
                if path_text in keep:
                    protected.append(path_text)
                    protected_reasons["kept-epoch"] += 1
                else:
                    if not path_text.startswith(required_prefix):
                        raise RuntimeError(f"out-of-scope checkpoint: {{path_text}}")
                    candidates.append(path_text)
                    if temporary:
                        candidate_reasons["temporary"] += 1
                    elif target_epoch < 1:
                        candidate_reasons["fractional-target"] += 1
                    elif int(target_epoch) % 2 == 1:
                        candidate_reasons["odd-target-or-intermediate"] += 1
                    elif target_epoch in (6, 10):
                        candidate_reasons["explicit-e6-e10"] += 1
                    else:
                        candidate_reasons["other-surplus"] += 1

        candidates = sorted(set(candidates))
        protected = sorted(set(protected))
        overlap = sorted(set(candidates) & set(protected))
        if overlap:
            raise RuntimeError(f"candidate/protected overlap: {{overlap[:5]}}")
        manifest_text = "\\n".join(candidates)
        manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()

        size_kib = 0
        for start in range(0, len(candidates), 100):
            chunk = candidates[start : start + 100]
            if not chunk:
                continue
            result = subprocess.run(["du", "-sk", *chunk], check=True, stdout=subprocess.PIPE, text=True)
            for line in result.stdout.splitlines():
                size_kib += int(line.split(None, 1)[0])

        summary = {{
            "mode": config["mode"],
            "root": str(root),
            "parentCount": len(parents),
            "candidateCount": len(candidates),
            "protectedCount": len(protected),
            "candidateSizeKiB": size_kib,
            "candidateSizeTiB": size_kib / (1024 ** 3),
            "manifestSha256": manifest_hash,
            "candidateReasons": dict(sorted(candidate_reasons.items())),
            "protectedReasons": dict(sorted(protected_reasons.items())),
            "activeOutputs": sorted(active_outputs),
            "excludedBatches": sorted(excluded_batches),
            "unparseableOutputCount": len(unparseable),
            "unparseableOutputs": unparseable,
            "candidateExamples": candidates[:20],
        }}
        print("CHECKPOINT_CLEANUP_SUMMARY=" + json.dumps(summary, separators=(",", ":")), flush=True)
        compressed = base64.b64encode(zlib.compress(manifest_text.encode(), 9)).decode()
        print("CHECKPOINT_CLEANUP_MANIFEST_ZLIB_BASE64=" + compressed, flush=True)

        if config["mode"] == "inventory":
            print("CHECKPOINT_CLEANUP_INVENTORY_COMPLETE", flush=True)
        else:
            expected_hash = config["expectedHash"]
            if manifest_hash != expected_hash:
                raise RuntimeError(f"manifest changed: expected {{expected_hash}}, got {{manifest_hash}}")
            deleted = []
            for path_text in candidates:
                path = Path(path_text)
                if not str(path.resolve()).startswith(required_prefix):
                    raise RuntimeError(f"resolved path escaped cleanup root: {{path}}")
                if str(path.parent) in active_outputs:
                    raise RuntimeError(f"candidate entered active output: {{path}}")
                if not checkpoint_pattern.fullmatch(path.name):
                    raise RuntimeError(f"candidate is not a checkpoint directory: {{path}}")
                shutil.rmtree(path)
                deleted.append(path_text)
            print(
                "CHECKPOINT_CLEANUP_DELETE_COMPLETE="
                + json.dumps({{"manifestSha256": manifest_hash, "deletedCount": len(deleted), "freedKiB": size_kib}}, separators=(",", ":")),
                flush=True,
            )
        """
    ).strip()


def build_spec(args: argparse.Namespace, base_spec: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    config = {
        "mode": args.mode,
        "root": args.root,
        "activeOutputs": sorted(set(args.protect_output)),
        "excludedBatches": sorted(set(args.exclude_batch)),
        "expectedHash": args.expected_hash,
    }
    program = remote_program(config)
    encoded_program = base64.b64encode(program.encode()).decode()
    command = f"import base64; exec(compile(base64.b64decode({encoded_program!r}), '<checkpoint-cleanup>', 'exec'))"
    task["name"] = "checkpoint-cleanup"
    task["command"] = ["python"]
    task["arguments"] = ["-c", command]
    task["resources"] = {"cpuCount": 2, "memory": "8 GiB"}
    task["context"] = {"priority": args.priority, "minRuntime": 0, "autoResume": False}
    task["propagateFailure"] = True
    task["propagatePreemption"] = False
    task["envVars"] = [
        env
        for env in task.get("envVars", [])
        if env.get("name") not in {"WANDB_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"}
    ]
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    spec = build_spec(args, get_base_spec(args.base_experiment))
    if args.print_only:
        json.dump(spec, __import__("sys").stdout, indent=2)
        print()
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(spec, handle)
        handle.flush()
        result = subprocess.run(
            [
                "beaker",
                "experiment",
                "create",
                handle.name,
                "--workspace",
                args.workspace,
                "--priority",
                args.priority,
                "--name",
                args.name,
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
