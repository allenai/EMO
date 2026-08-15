#!/usr/bin/env python3
"""Conservatively identify surplus numeric checkpoints in an ICSL Weka tree.

The caller supplies report-derived references as base64-encoded JSON. This
script never deletes anything. It keeps explicit references, every checkpoint
in active or recently modified output directories, the latest checkpoint in
each parseable completed output, and integer-epoch WSD stable/final steps.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


STEP_RE = re.compile(r"step(?P<step>[0-9]+)$")
EPOCH_RE = re.compile(r"_e(?P<epoch>[0-9]+)(?:_|$)")
BS_RE = re.compile(r"_bs(?P<batch>64|256|512|1024)(?:_|$)")
WARMUP_RE = re.compile(r"_warmup(?P<warmup>[0-9]+)(?:_|$)")
WARMUP_BATCH = {384: 64, 96: 256, 95: 256, 48: 512, 24: 1024}


def category(name: str) -> str:
    if name.startswith("dense_1b_step0_"):
        return "dense-1b-cosine"
    if name.startswith("dense_1b_step1_"):
        return "dense-1b-step1"
    if name.startswith("dense_1b_step2_1_"):
        return "dense-1b-step2-1"
    if name.startswith("dense_1b_step2_"):
        return "dense-1b-step2"
    if name.startswith(("dense_153m_", "dense_474m_")):
        return "dense-small"
    if name.startswith("stdmoe_"):
        return "moe"
    return "other"


def run_shape(name: str) -> tuple[int, int] | None:
    epoch_match = EPOCH_RE.search(name)
    if epoch_match is None:
        return None
    batch_match = BS_RE.search(name)
    if batch_match is not None:
        batch = int(batch_match.group("batch"))
    else:
        warmup_match = WARMUP_RE.search(name)
        if warmup_match is None:
            return None
        batch = WARMUP_BATCH.get(int(warmup_match.group("warmup")), 0)
    if not batch:
        return None
    return int(epoch_match.group("epoch")), batch


def inventory(models: Path) -> dict[Path, list[tuple[int, Path, float]]]:
    grouped: dict[Path, list[tuple[int, Path, float]]] = defaultdict(list)
    for directory, children, _ in os.walk(models):
        parent = Path(directory)
        kept_children: list[str] = []
        for child in children:
            match = STEP_RE.fullmatch(child)
            if match is None:
                kept_children.append(child)
                continue
            path = parent / child
            grouped[parent].append((int(match.group("step")), path, path.stat().st_mtime))
        children[:] = kept_children
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--refs-base64", required=True)
    parser.add_argument("--recent-hours", type=float, default=2.0)
    parser.add_argument("--emit-paths", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    models = root / "models"
    required_prefix = str(models) + "/"
    refs = json.loads(base64.b64decode(args.refs_base64).decode())
    explicit = {str(Path(path)) for path in refs.get("checkpointPaths", [])}
    active_outputs = {str(Path(path)) for path in refs.get("activeOutputs", [])}
    baseline = {str(Path(path)) for path in refs.get("baselineCandidates", [])}
    grouped = inventory(models)
    cutoff = time.time() - args.recent_hours * 3600

    candidates: list[str] = []
    protected: dict[str, str] = {}
    parents_by_category: Counter[str] = Counter()
    candidates_by_category: Counter[str] = Counter()
    protected_by_reason: Counter[str] = Counter()
    unparseable_parents: list[str] = []

    for parent, entries in grouped.items():
        name = parent.name
        kind = category(name)
        parents_by_category[kind] += 1
        entries.sort()
        shape = run_shape(name)
        is_recent = max(mtime for _, _, mtime in entries) >= cutoff
        is_active = str(parent) in active_outputs
        keep: dict[str, str] = {}

        for _, path, _ in entries:
            path_text = str(path)
            if path_text in explicit:
                keep[path_text] = "report-or-config-reference"

        if is_active:
            for _, path, _ in entries:
                keep[str(path)] = "active-output"
        elif is_recent:
            for _, path, _ in entries:
                keep[str(path)] = "recent-output"
        elif shape is None:
            unparseable_parents.append(str(parent))
            for _, path, _ in entries:
                keep[str(path)] = "unparseable-output"
        else:
            target_epoch, batch = shape
            # Retain the latest observed step as a conservative final/partial
            # endpoint even when an old job exited before its intended target.
            keep[str(entries[-1][1])] = "latest-step"
            raw_steps_per_epoch = 1_000_000_000 / (batch * 4096)
            is_wsd = "_wsd_" in name
            expected: set[int] = set()
            for epoch in range(1, target_epoch + 1):
                final_value = epoch * raw_steps_per_epoch
                expected.update(
                    {
                        math.floor(final_value),
                        round(final_value),
                        math.ceil(final_value),
                    }
                )
                if is_wsd:
                    stable_value = 0.9 * final_value
                    expected.update(
                        {
                            math.floor(stable_value),
                            round(stable_value),
                            math.ceil(stable_value),
                        }
                    )
            for step, path, _ in entries:
                if any(abs(step - value) <= 1 for value in expected):
                    keep.setdefault(str(path), "integer-epoch-stable-or-final")

        for _, path, _ in entries:
            path_text = str(path)
            if not path_text.startswith(required_prefix):
                raise RuntimeError(f"out-of-scope checkpoint: {path_text}")
            if path_text in keep:
                protected[path_text] = keep[path_text]
                protected_by_reason[keep[path_text]] += 1
            else:
                candidates.append(path_text)
                candidates_by_category[kind] += 1

    candidates.sort()
    if len(candidates) != len(set(candidates)):
        raise RuntimeError("duplicate candidate paths")
    overlap = sorted(set(candidates) & explicit)
    if overlap:
        raise RuntimeError(f"candidate/reference overlap: {overlap[:5]}")

    summary: dict[str, object] = {
        "root": str(root),
        "numericCheckpointCount": sum(len(entries) for entries in grouped.values()),
        "outputDirectoryCount": len(grouped),
        "candidateCount": len(candidates),
        "protectedCount": len(protected),
        "candidateByCategory": dict(sorted(candidates_by_category.items())),
        "outputDirectoriesByCategory": dict(sorted(parents_by_category.items())),
        "protectedByReason": dict(sorted(protected_by_reason.items())),
        "explicitReferenceCount": len(explicit),
        "activeOutputCount": len(active_outputs),
        "unparseableOutputCount": len(unparseable_parents),
        "baselineCandidateCount": len(baseline),
        "baselineNowProtected": len(baseline & set(protected)),
        "baselineStillCandidate": len(baseline & set(candidates)),
        "recentHoursProtected": args.recent_hours,
        "candidateExamples": candidates[:20],
    }
    if args.emit_paths:
        summary["candidates"] = candidates
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
