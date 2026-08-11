#!/usr/bin/env python3
"""Audit the effective weight decay of nested Pool-3B continuations.

Pool-3B runs restore optimizer state.  For historical runs created before the
matched-source fix, the optimizer's weight decay therefore came from the root
1B-pool checkpoint and propagated through every continuation, regardless of
the weight decay encoded in the Pool-3B run name.  This tool reconstructs that
ancestry without touching Beaker or Weka and emits a machine-readable audit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


REPORTS = {
    "153m": Path("reports/0802/data/wsd_batch_size_153m_pool3b.json"),
    "474m": Path("reports/0802/data/wsd_batch_size_474m_pool3b.json"),
    "1b": Path("reports/0802/data/wsd_batch_size_1b_pool3b.json"),
}
WD_RE = re.compile(r"_wd([^_/]+)_warmup")
STEP_RE = re.compile(r"^(?P<output>.+)/step\d+$")


def decimal_text(value: Any) -> str:
    return format(Decimal(str(value)), "f")


def checkpoint_output(path: str | None) -> str | None:
    if not path:
        return None
    match = STEP_RE.fullmatch(path)
    return match.group("output") if match else None


def named_wd(path: str | None) -> str | None:
    if not path:
        return None
    match = WD_RE.search(path)
    return decimal_text(match.group(1)) if match else None


@dataclass
class Run:
    model: str
    report: Path
    index: int
    sweep: dict[str, Any]
    effective_wd: str | None = None

    @property
    def output(self) -> str | None:
        return self.sweep.get("output")

    @property
    def source_checkpoint(self) -> str | None:
        return self.sweep.get("sourceCheckpoint")

    @property
    def intended_wd(self) -> str:
        return decimal_text(self.sweep["wd"])

    @property
    def epoch(self) -> int:
        return int(self.sweep["activeEpoch"])

    @property
    def batch(self) -> int:
        return int(self.sweep["batchSequences"])

    @property
    def lr(self) -> str:
        return decimal_text(self.sweep["lr"])

    @property
    def result(self) -> dict[str, Any] | None:
        result = self.sweep.get("results", {}).get(str(self.epoch))
        return result if isinstance(result, dict) else None

    @property
    def completed_endpoint(self) -> bool:
        return bool(self.result and self.result.get("status") == "complete")


def load_runs() -> tuple[list[Run], dict[Path, dict[str, Any]]]:
    runs: list[Run] = []
    reports: dict[Path, dict[str, Any]] = {}
    for model, path in REPORTS.items():
        report = json.loads(path.read_text())
        reports[path] = report
        runs.extend(
            Run(model=model, report=path, index=index, sweep=sweep)
            for index, sweep in enumerate(report.get("batchSweeps", []))
        )
    return runs, reports


def resolve_effective_wds(runs: Iterable[Run]) -> None:
    runs = list(runs)
    by_output = {run.output: run for run in runs if run.output}
    unresolved = set(range(len(runs)))
    while unresolved:
        progress = False
        for index in tuple(unresolved):
            run = runs[index]
            source_output = checkpoint_output(run.source_checkpoint)
            source_run = by_output.get(source_output)
            if source_run is run:
                # An infrastructure recovery can restore an exact checkpoint
                # inside its own retained output directory.  After canonical
                # consolidation that is an intentional self-reference, so the
                # corrected directory name is the effective-WD authority.
                run.effective_wd = named_wd(run.source_checkpoint)
            elif source_run is not None:
                if source_run.effective_wd is None:
                    continue
                run.effective_wd = source_run.effective_wd
            else:
                run.effective_wd = named_wd(run.source_checkpoint)
            unresolved.remove(index)
            progress = True
        if not progress:
            details = [
                {
                    "model": runs[index].model,
                    "beaker": runs[index].sweep.get("beaker"),
                    "sourceCheckpoint": runs[index].source_checkpoint,
                }
                for index in sorted(unresolved)
            ]
            raise RuntimeError(f"unable to resolve effective WD ancestry: {details}")


def canonical_score(run: Run) -> tuple[int, int, int, int]:
    # Prefer a complete endpoint, then an already-correct directory name, then
    # a successful sweep, and finally the latest registered attempt.
    return (
        int(run.completed_endpoint),
        int(run.intended_wd == run.effective_wd),
        int(run.sweep.get("status") == "complete"),
        run.index,
    )


def build_audit(runs: list[Run]) -> dict[str, Any]:
    resolve_effective_wds(runs)
    groups: dict[tuple[str, int, int, str, str | None], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.model, run.batch, run.epoch, run.lr, run.effective_wd)].append(run)

    referenced_outputs = {
        output for run in runs if (output := checkpoint_output(run.source_checkpoint)) is not None
    }
    duplicate_groups = []
    cleanup_groups = []
    canonical_outputs: set[str] = set()
    redundant_outputs: set[str] = set()
    for key, members in sorted(groups.items()):
        completed = [run for run in members if run.completed_endpoint]
        referenced = [run for run in members if run.output and run.output in referenced_outputs]
        canonical = (
            max(completed, key=canonical_score)
            if completed
            else max(referenced, key=lambda run: run.index)
            if referenced
            else members[0]
            if len(members) == 1
            else None
        )
        member_outputs = [run.output for run in members if run.output]
        redundant_members = [run for run in members if run is not canonical]
        rename_output = None
        if (
            canonical is not None
            and canonical.output
            and canonical.intended_wd != canonical.effective_wd
        ):
            corrected_name = WD_RE.sub(
                f"_wd{canonical.effective_wd}_warmup", canonical.output, count=1
            )
            if corrected_name == canonical.output:
                raise RuntimeError(f"failed to construct corrected output for {canonical.output}")
            rename_output = corrected_name
        cleanup_groups.append(
            {
                "coordinate": {
                    "model": key[0],
                    "batchSequences": key[1],
                    "epoch": key[2],
                    "lr": key[3],
                    "effectiveWd": key[4],
                },
                "canonicalBeaker": canonical.sweep.get("beaker") if canonical else None,
                "canonicalOutput": canonical.output if canonical else None,
                "correctedCanonicalOutput": rename_output
                or (canonical.output if canonical else None),
                "consolidateFromOutputs": [run.output for run in redundant_members if run.output],
                "deleteOutputs": [run.output for run in redundant_members if run.output],
                "memberOutputs": member_outputs,
            }
        )
        if not completed:
            continue
        if canonical.output:
            canonical_outputs.add(canonical.output)
        redundant = [run for run in completed if run is not canonical]
        redundant_outputs.update(run.output for run in redundant if run.output)
        if redundant:
            duplicate_groups.append(
                {
                    "coordinate": {
                        "model": key[0],
                        "batchSequences": key[1],
                        "epoch": key[2],
                        "lr": key[3],
                        "effectiveWd": key[4],
                    },
                    "canonical": canonical.sweep.get("beaker"),
                    "canonicalOutput": canonical.output,
                    "redundant": [run.sweep.get("beaker") for run in redundant],
                    "redundantOutputs": [run.output for run in redundant],
                }
            )

    entries = []
    for run in runs:
        result = run.result or {}
        entries.append(
            {
                "model": run.model,
                "report": str(run.report),
                "reportIndex": run.index,
                "batchSequences": run.batch,
                "epoch": run.epoch,
                "lr": run.lr,
                "intendedWd": run.intended_wd,
                "effectiveWd": run.effective_wd,
                "wdMismatch": run.intended_wd != run.effective_wd,
                "status": run.sweep.get("status"),
                "endpointStatus": result.get("status"),
                "validation": result.get("validation"),
                "beaker": run.sweep.get("beaker"),
                "recoveryOf": run.sweep.get("recoveryOf"),
                "output": run.output,
                "sourceCheckpoint": run.source_checkpoint,
                "resumeCheckpoint": result.get("resumeCheckpoint"),
                "outputReferencedAsSource": run.output in referenced_outputs,
                "canonicalCompletedOutput": run.output in canonical_outputs,
                "redundantCompletedOutput": run.output in redundant_outputs,
            }
        )

    return {
        "schemaVersion": 1,
        "summary": {
            "registeredRuns": len(runs),
            "completedEndpoints": sum(run.completed_endpoint for run in runs),
            "wdMismatches": sum(run.intended_wd != run.effective_wd for run in runs),
            "duplicateCompletedGroups": len(duplicate_groups),
            "canonicalCompletedOutputs": len(canonical_outputs),
            "redundantCompletedOutputs": len(redundant_outputs),
        },
        "duplicateGroups": duplicate_groups,
        "cleanupPlan": {
            "expectedReportedOutputs": sorted(run.output for run in runs if run.output),
            "groups": cleanup_groups,
        },
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs, _ = load_runs()
    audit = build_audit(runs)
    payload = json.dumps(audit, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
