#!/usr/bin/env python3
"""Consolidate and rename historical Pool-3B outputs after the WD audit.

This is deliberately a two-phase maintenance tool.  Without ``--apply`` it
performs a strict dry-run and prints every directory that would be removed or
renamed.  With ``--apply`` it first moves every uniquely retained ``step*``
checkpoint into the selected canonical output, validates colliding checkpoint
layouts, removes redundant outputs, corrects the saved checkpoint config, and
finally renames the canonical output to show its effective weight decay.

The input is produced by ``audit_dense_pool3b_effective_wd.py``.  The tool is
intended to run in a CPU-only Beaker maintenance session with Weka mounted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"^step\d+$")


def checkpoint_layout(path: Path) -> list[tuple[str, int]]:
    """Return the model/optimizer tensor-file layout for a checkpoint.

    ``config.json``, distributed-checkpoint ``.metadata``, and trainer rank
    state legitimately encode the originally requested WD, output path, and
    per-run RNG/progress metadata.  The tensor shard layout is the relevant
    redundancy check here: the optimizer state load made both historical WD
    branches execute the same effective optimizer coordinate.
    """
    return sorted(
        (str(item.relative_to(path)), item.stat().st_size)
        for item in (path / "model_and_optim").rglob("*")
        if item.is_file() and item.name != ".metadata"
    )


def set_effective_wd(config_path: Path, effective_wd: str) -> None:
    config = json.loads(config_path.read_text())
    train_module = config.get("train_module")
    if not isinstance(train_module, dict):
        raise RuntimeError(f"missing train_module config in {config_path}")
    optim = train_module.get("optim")
    if not isinstance(optim, dict) or "weight_decay" not in optim:
        raise RuntimeError(f"missing train_module.optim.weight_decay in {config_path}")
    historical = optim["weight_decay"]
    optim["weight_decay"] = float(effective_wd)
    config["pool3b_weight_decay_correction"] = {
        "historical_intended_weight_decay": historical,
        "effective_weight_decay": float(effective_wd),
        "reason": (
            "Historical Pool-3B continuation loaded optimizer weight_decay "
            "from its root 1B-pool checkpoint."
        ),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/weka/oe-training-default/sewonm/icsl/models"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    audit: dict[str, Any] = json.loads(args.audit.read_text())
    plan = audit["cleanupPlan"]
    reported = {Path(path) for path in plan["expectedReportedOutputs"]}
    actual = {path for path in root.glob("*pool3b*") if path.is_dir()}
    unexpected = sorted(actual - reported)
    if unexpected:
        raise RuntimeError(
            "unreported Pool-3B directories make cleanup unsafe:\n"
            + "\n".join(str(path) for path in unexpected)
        )

    consolidations: list[tuple[Path, Path]] = []
    duplicate_steps: list[tuple[Path, Path]] = []
    planned_targets: dict[Path, Path] = {}
    deletions: set[Path] = set()
    renames: list[tuple[Path, Path]] = []
    corrections: list[tuple[Path, str]] = []

    for group in plan["groups"]:
        coordinate = group["coordinate"]
        effective_wd = coordinate["effectiveWd"]
        canonical_value = group["canonicalOutput"]
        corrected_value = group["correctedCanonicalOutput"]
        sources = [Path(path) for path in group["consolidateFromOutputs"]]

        if canonical_value is None:
            deletions.update(path for path in sources if path.exists())
            continue

        canonical = Path(canonical_value)
        corrected = Path(corrected_value)
        if not canonical.exists():
            if corrected.exists() and corrected != canonical:
                canonical = corrected
            elif any(path.exists() for path in sources):
                raise RuntimeError(
                    f"canonical output is missing while redundant outputs exist: {canonical_value}"
                )
            else:
                continue

        for source in sources:
            if not source.exists() or source == canonical:
                continue
            for step in sorted(source.iterdir()):
                if not step.is_dir() or not STEP_RE.fullmatch(step.name):
                    continue
                target = canonical / step.name
                if target.exists():
                    if checkpoint_layout(step) != checkpoint_layout(target):
                        raise RuntimeError(
                            f"colliding checkpoints are not layout-identical: {step} vs {target}"
                        )
                    duplicate_steps.append((step, target))
                elif target in planned_targets:
                    planned_source = planned_targets[target]
                    if checkpoint_layout(step) != checkpoint_layout(planned_source):
                        raise RuntimeError(
                            "multiple sources for a missing canonical checkpoint are not "
                            f"layout-identical: {step} vs {planned_source}"
                        )
                    duplicate_steps.append((step, target))
                else:
                    consolidations.append((step, target))
                    planned_targets[target] = step
            deletions.add(source)

        final_path = corrected if corrected != Path(canonical_value) else canonical
        if corrected != Path(canonical_value) and canonical != corrected:
            if corrected.exists() and corrected not in deletions:
                raise RuntimeError(f"rename destination already exists: {corrected}")
            renames.append((canonical, corrected))
        corrections.append((final_path, effective_wd))

    # Never touch paths outside the expected Pool-3B namespace.
    touched = (
        [path for pair in consolidations for path in pair]
        + [path for pair in duplicate_steps for path in pair]
        + list(deletions)
        + [path for pair in renames for path in pair]
        + [path for path, _ in corrections]
    )
    for path in touched:
        resolved = path.resolve(strict=False)
        if root not in resolved.parents or "pool3b" not in str(resolved):
            raise RuntimeError(f"refusing unsafe path: {path}")

    remaining = len(actual - deletions)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "actualPool3bDirectories": len(actual),
                "missingReportedDirectories": len(reported - actual),
                "consolidateUniqueSteps": len(consolidations),
                "discardDuplicateSteps": len(duplicate_steps),
                "deleteDirectories": len(deletions),
                "renameDirectories": len(renames),
                "expectedRemainingDirectories": remaining,
                "deletions": [str(path) for path in sorted(deletions)],
                "renames": [[str(source), str(target)] for source, target in renames],
            },
            indent=2,
        )
    )
    if not args.apply:
        return

    for source, target in consolidations:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
    for path in sorted(deletions):
        if path.exists():
            shutil.rmtree(path)
    for source, target in renames:
        os.rename(source, target)
    corrected_configs = 0
    for output, effective_wd in corrections:
        if not output.exists():
            continue
        for config_path in output.glob("step*/config.json"):
            set_effective_wd(config_path, effective_wd)
            corrected_configs += 1

    final_actual = {path for path in root.glob("*pool3b*") if path.is_dir()}
    if len(final_actual) != remaining:
        raise RuntimeError(
            f"post-cleanup directory count mismatch: expected {remaining}, got {len(final_actual)}"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "remainingDirectories": len(final_actual),
                "correctedCheckpointConfigs": corrected_configs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
