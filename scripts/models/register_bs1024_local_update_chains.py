#!/usr/bin/env python3
"""Register the five authorized BS1024 local-update chain experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from submit_bs1024_local_update_chain import (
    DILOCO_NAME,
    LOCAL_NAMES,
    conventional_source,
    local_native_start,
    output_path,
)
from submit_bs1024_local_update_frontier import FRONTIERS

REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
MIRROR = Path("reports/0802/data/wsd_batch_simulation_1b.js")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full SHA")
    return args


def method_key(method: str, init: int, sim: int) -> str:
    if method == "local_sgd":
        return f"post_e{init}_local_sgd_h4_bs1024_dr_simbs{sim}"
    return f"post_e{init}_diloco_h32_olr0.5_om0.7_vrecal_bs1024_dr_simbs64"


def ensure_column(report: dict, *, method: str, init: int, sim: int) -> None:
    key = method_key(method, init, sim)
    if any(column.get("key") == key for column in report["columns"]):
        return
    report["columns"].append(
        {
            "key": key,
            "label": (
                f"DiLoCo (BS1024, H=32, simBS64, outer0.5/momentum0.7, v-recal, "
                f"BS1024 E{init} init, DR)"
            ),
            "tableLabel": (
                "DiLoCo\nBS1024\nouterLR=0.5\nmomentum=0.7\nH=32\n"
                f"simBS64\nv-recal\nBS1024 E{init} init\nDR"
            ),
            "globalBatchSequences": 1024,
            "simulatedBatchSequences": 64,
            "initialization": f"BS1024 DR E{init} exact pre-decay checkpoint",
            "syncInterval": 32,
            "outerLearningRate": 0.5,
            "outerMomentum": 0.7,
            "secondMomentRecalibration": True,
            "secondMomentRecalibrationRatio": 16,
            "dynamicRepacking": True,
        }
    )


def plan(
    *, method: str, sim: int, init: int, wd: str, chain_name: str, revision: str
) -> dict:
    output = output_path(method, sim, init, wd)
    if method == "diloco" or wd == "1.0":
        start = init
        source = conventional_source(init, wd)
        recalibrate = True
    else:
        start, source = local_native_start(sim, init)
        recalibrate = False
    ladder = [target for target in (4, 8, 12, 16) if target > start]
    return {
        "planId": f"{chain_name}-wd{wd}",
        "chainName": chain_name,
        "method": method_key(method, init, sim),
        "syncInterval": 4 if method == "local_sgd" else 32,
        "startEpoch": start,
        "targetEpoch": 16,
        "targetLadder": ladder,
        "batchSequences": 1024,
        "simulatedBatchSequences": sim,
        "lr": "1e-3",
        "wd": wd,
        "status": "planned",
        "healthStatus": "planned",
        "currentStep": FRONTIERS[start][0],
        "revision": revision,
        "sourceCheckpoint": source,
        "output": output,
        "preDecayCheckpoint": f"{output}/step3432",
        "endpointCheckpoint": f"{output}/step3815",
        "recalibrateSecondMomentOnStart": recalibrate,
        "secondMomentRecalibrationRatio": 1024 // sim,
        "dilocoOuterLr": 0.5 if method == "diloco" else None,
        "dilocoOuterMomentum": 0.7 if method == "diloco" else None,
        "wdGatePolicy": (
            "At each matched healthy frontier, prune WD0.333 for later epochs only if WD1.0 has lower validation CE."
            if sim == 256
            else None
        ),
        "results": {},
        "reason": (
            "Registered as one persistent four-node chain through E16. Every evaluated stage performs terminal decay, while each continuation loads the preceding exact pre-decay checkpoint."
        ),
    }


def compact_none(value: object) -> object:
    if isinstance(value, dict):
        return {key: compact_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [compact_none(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    report = json.loads(REPORT.read_text())
    report["updated"] = "2026-08-15"
    report["setup"] = (
        "Dense 1B; sealed repeated 1B-token DCLM pool; sequence length4096; global BS1024; "
        "dynamic repacking; LR1e-3; rank microbatch4; token-based WSD with10% terminal decay. "
        "The active persistent-chain program runs LocalSGD H4 over simBS64/256 and E2/E4 "
        "initialization through E16, plus DiLoCo H32 simBS64 outerLR0.5/momentum0.7 over both "
        "initializations through E16."
    )
    report["selection"] = {
        "targetEpoch": 16,
        "learningRate": "1e-3",
        "localSgdSyncInterval": 4,
        "dilocoSyncInterval": 32,
        "dilocoOuterLearningRate": 0.5,
        "dilocoOuterMomentum": 0.7,
        "criterion": (
            "Evaluate every frontier after terminal decay; continue only from exact pre-decay state. "
            "For simBS256, compare WD0.333 and1.0 at matched frontiers and prune WD0.333 later only when WD1.0 wins."
        ),
    }
    for init in (2, 4):
        ensure_column(report, method="diloco", init=init, sim=64)
    requested: list[dict] = []
    for (sim, init), chain_name in LOCAL_NAMES.items():
        wds = ("0.333",) if sim == 64 else ("0.333", "1.0")
        for wd in wds:
            requested.append(plan(
                method="local_sgd", sim=sim, init=init, wd=wd,
                chain_name=chain_name, revision=args.revision,
            ))
    for init in (2, 4):
        requested.append(plan(
            method="diloco", sim=64, init=init, wd="0.333",
            chain_name=DILOCO_NAME, revision=args.revision,
        ))
    chain_names = set(LOCAL_NAMES.values()) | {DILOCO_NAME}
    existing = [run for run in report["runs"] if run.get("chainName") in chain_names]
    if existing:
        raise RuntimeError("one or more requested chains are already registered")
    report["runs"].extend(compact_none(requested))
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    MIRROR.write_text(
        "window.ICSL_BATCH_SIMULATION_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )
    print(f"registered {len(requested)} trajectory records across five chains")


if __name__ == "__main__":
    main()
