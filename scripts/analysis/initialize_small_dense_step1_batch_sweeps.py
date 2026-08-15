#!/usr/bin/env python3
"""Initialize the 153M/474M Step 1-1 reports for BS64 through BS512."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports/0802/data"
TARGETS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
BATCHES = [64, 128, 256, 512]
SPECS = {
    "153m": "Dense 153M",
    "474m": "Dense 474M",
}


for slug, label in SPECS.items():
    path = DATA / f"wsd_batch_size_{slug}.json"
    report = json.loads(path.read_text())
    report["updated"] = "2026-08-09"
    report["title"] = f"0802 Step 1-1 · {label}"
    report["setup"] = (
        f"{label}; 1B-unique-pool repeated-token regime; sequence length 4096; "
        "adaptive global batches 64, 128, 256, and 512 sequences. Rank microbatch "
        "is 16 sequences; BS64 uses 4 GPUs and BS128/256/512 use 8 GPUs with "
        "gradient accumulation 1/1/2/4. Warmup is token-matched across batches."
    )
    report["selection"] = (
        "At each frontier, select the lowest healthy DCLM validation CE. E1 starts "
        "from transferred 1B WD caps and one lower rung with LR 1e-3/2e-3 axial "
        "neighbors. Later WD may not decrease; exact-coordinate resumes only. Stop "
        "a batch after its selected validation CE no longer improves. Historical "
        "runs remain visible and reusable when their exact numeric tuple matches."
    )
    report["selectionPolicy"] = {"allowAllCompletedCoordinates": True}
    report["targetEpochs"] = TARGETS
    report["batchTargetEpochs"] = {str(batch): TARGETS for batch in BATCHES}
    report["summaryBatches"] = BATCHES
    report["gpuTopology"] = {
        "rankMicrobatchSequences": 16,
        "64": {"gpuCount": 4, "gradientAccumulation": 1},
        "128": {"gpuCount": 8, "gradientAccumulation": 1},
        "256": {"gpuCount": 8, "gradientAccumulation": 2},
        "512": {"gpuCount": 8, "gradientAccumulation": 4},
    }
    text = json.dumps(report, indent=2) + "\n"
    path.write_text(text)
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )
