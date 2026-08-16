#!/usr/bin/env python3
"""Register the four E4-init sequential-replay window-4 ablations."""

from __future__ import annotations

import json
from pathlib import Path


REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
MIRROR = Path("reports/0802/data/wsd_batch_simulation_1b.js")
REVISION = "0d0679918f5589f5b76f0b4a90f25db4f6db1a3d"
SOURCE = (
    "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/"
    "bs1024_dr_lr1e-3_wd0.333/step858"
)
ROOT = "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b"


METHODS = (
    {
        "key": "post_e4_sequential_replay_bs1024_dr_simbs256",
        "label": "Sequential replay (BS1024, simBS256, BS1024 E4 init, DR)",
        "tableLabel": "Sequential replay\nBS1024\nsimBS256\nBS1024 E4 init\nDR",
        "globalBatchSequences": 1024,
        "simulatedBatchSequences": 256,
        "initialization": "BS1024 DR E4 exact pre-decay checkpoint",
        "replayPackets": 4,
        "maximumGradientStaleness": 3,
        "secondMomentRecalibration": True,
        "secondMomentRecalibrationRatio": 4,
        "dynamicRepacking": True,
    },
    {
        "key": "post_e4_sequential_replay_bs512_dr_simbs128",
        "label": "Sequential replay (BS512, simBS128, BS1024 E4 init, DR)",
        "tableLabel": "Sequential replay\nBS512\nsimBS128\nBS1024 E4 init\nDR",
        "globalBatchSequences": 512,
        "simulatedBatchSequences": 128,
        "initialization": "BS1024 DR E4 exact pre-decay checkpoint",
        "replayPackets": 4,
        "maximumGradientStaleness": 3,
        "secondMomentRecalibration": True,
        "secondMomentRecalibrationRatio": 4,
        "dynamicRepacking": True,
    },
)


def output_path(global_bs: int, sim_bs: int, lr: str) -> str:
    return (
        f"{ROOT}/bs{global_bs}_dr_sequential_replay_simbs{sim_bs}_"
        f"init=bs1024e4_lr1e-3_runlr{lr}_wd0.333"
    )


def run(global_bs: int, sim_bs: int, lr: str) -> dict:
    method = f"post_e4_sequential_replay_bs{global_bs}_dr_simbs{sim_bs}"
    lr_id = "1e-3" if lr == "1e-3" else "5e-4"
    pre_decay, endpoint = (1716, 1908) if global_bs == 1024 else (2574, 2957)
    output = output_path(global_bs, sim_bs, lr)
    return {
        "planId": (
            f"bs{global_bs}-e4init-sequential-replay-sim{sim_bs}-"
            f"lr{lr_id}-wd0.333-e8"
        ),
        "method": method,
        "startEpoch": 4,
        "targetEpoch": 8,
        "batchSequences": global_bs,
        "simulatedBatchSequences": sim_bs,
        "lr": lr,
        "wd": "0.333",
        "status": "planned",
        "healthStatus": "planned",
        "revision": REVISION,
        "sourceCheckpoint": SOURCE,
        "output": output,
        "preDecayCheckpoint": f"{output}/step{pre_decay}",
        "endpointCheckpoint": f"{output}/step{endpoint}",
        "replayPackets": 4,
        "maximumGradientStaleness": 3,
        "recalibrateSecondMomentOnStart": True,
        "secondMomentRecalibrationRatio": 4,
        "sequentialReplayMicrobatchGradients": False,
        "schedulerUnits": "tokens",
        "results": {},
        "reason": (
            "Authorized E4-init to E8 sequential-replay window-4 ablation. "
            "Registered before submission; exact conventional step858 source, "
            "token-matched WSD frontier, and one-time Adam-v recalibration ratio4."
        ),
    }


def main() -> None:
    report = json.loads(REPORT.read_text())
    report["updated"] = "2026-08-15"
    report["setup"] = (
        "Dense 1B; sealed 1B-token training pool; sequence length4096; dynamic "
        "repacking; token-based WSD with10% terminal decay. The active ablation "
        "compares sequential replica-gradient replay with four packets and maximum "
        "staleness3 at BS1024/simBS256 and BS512/simBS128, each at LR1e-3 and5e-4, "
        "from the exact conventional BS1024 DR E4 pre-decay checkpoint through E8."
    )
    report["selection"] = {
        "targetEpoch": 8,
        "learningRates": ["1e-3", "5e-4"],
        "weightDecay": "0.333",
        "criterion": (
            "Compare train CE, held-out DCLM validation CE, gap, downstream metrics, "
            "and avg8 BPB at matched E8 tokens across the four replay-window4 runs."
        ),
    }

    columns = report["columns"]
    columns[:] = [
        column
        for column in columns
        if column["key"] != "post_e4_sequential_replay_bs256_dr_simbs64"
    ]
    existing_methods = {column["key"] for column in columns}
    for method in METHODS:
        if method["key"] not in existing_methods:
            columns.append(method)

    runs = report["runs"]
    runs[:] = [
        entry
        for entry in runs
        if not (
            entry.get("status") == "planned"
            and entry.get("startEpoch") == 4
            and "sequential-replay" in entry.get("planId", "")
        )
    ]
    additions = [
        run(1024, 256, "1e-3"),
        run(1024, 256, "5e-4"),
        run(512, 128, "1e-3"),
        run(512, 128, "5e-4"),
    ]
    existing_plans = {entry["planId"] for entry in runs if "planId" in entry}
    duplicates = [entry["planId"] for entry in additions if entry["planId"] in existing_plans]
    if duplicates:
        raise RuntimeError(f"refusing duplicate plan IDs: {duplicates}")
    runs.extend(additions)

    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    MIRROR.write_text(
        "window.ICSL_BATCH_SIMULATION_DATA="
        + json.dumps(report, separators=(",", ":"))
        + ";\n"
    )


if __name__ == "__main__":
    main()
