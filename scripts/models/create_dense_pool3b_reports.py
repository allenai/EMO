#!/usr/bin/env python3
"""Create isolated 3B-pool Step 1 report registries from the 1B reports."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "0802"
DATA_DIR = REPORT_DIR / "data"
TARGETS = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88]
POOL_ROOT = "/weka/oe-training-default/sewonm/icsl/data/dclm_0802_nested_1b_3b_9b"

PLANS = {
    "153m": {
        "label": "153M",
        "rankMb": 16,
        "lr": {"64": "2e-3", "128": "2e-3", "256": "2e-3"},
        "initialWd": {"64": ["0.1", "0.033"], "128": ["0.1", "0.033"], "256": ["0.033", "0.01"]},
    },
    "474m": {
        "label": "474M",
        "rankMb": 16,
        "lr": {"64": "2e-3", "128": "2e-3", "256": "2e-3"},
        "initialWd": {"64": ["0.1", "0.033"], "128": ["0.1", "0.033"], "256": ["0.1", "0.033"]},
    },
    "1b": {
        "label": "1.5B",
        "rankMb": 8,
        "lr": {"64": "1e-3", "128": "1e-3", "256": "1e-3"},
        "initialWd": {"64": ["0.3", "0.1"], "128": ["0.3", "0.1"], "256": ["0.333", "0.1"]},
    },
}


def write_json_and_js(path: Path, report: dict) -> None:
    payload = json.dumps(report, indent=2) + "\n"
    path.write_text(payload)
    path.with_suffix(".js").write_text(
        "window.ICSL_REPORT_DATA=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )


def main() -> None:
    for model, plan in PLANS.items():
        source_json = DATA_DIR / f"wsd_batch_size_{model}.json"
        source_html = REPORT_DIR / f"wsd_batch_size_{model}.html"
        source = json.loads(source_json.read_text())
        report = deepcopy(source)
        report["updated"] = "2026-08-09"
        report["title"] = f"0802 Step 1-1 · Dense {plan['label']} · nested 3B pool"
        report["setup"] = (
            f"Dense {plan['label']}; nested 3B pool = sealed 1B + disjoint new 2B; "
            "sequence length 4096; global batches 64, 128, and 256 sequences."
        )
        report["selection"] = (
            "LR is frozen to the selected 1B-pool value for the same model and batch. "
            "At E1 branch the exact selected 1B pre-decay model/optimizer checkpoint: "
            "continue once at the selected WD and once with WD lowered by one ladder step. "
            "At later frontiers compare the selected WD "
            "with at most one higher ladder step, capped by the same-epoch 1B WD. Stop "
            "on validation-CE non-improvement or at the same-model/same-batch 1B optimum."
        )
        report["baseline1b"] = {
            "source": str(source_json.relative_to(ROOT)),
            "batchSweeps": source.get("batchSweeps", []),
            "healthAudit": source.get("healthAudit", {}),
        }
        report["batchSweeps"] = []
        report["targetEpochs"] = TARGETS
        report["batchTargetEpochs"] = {batch: TARGETS for batch in ("64", "128", "256")}
        report["summaryBatches"] = [64, 128, 256]
        report["optimizerStepComparisons"] = []
        report.pop("pausedBatches", None)
        report["gpuTopology"] = {
            batch: {
                "gpuCountPerNode": min(int(batch) // plan["rankMb"], 8),
                "nodeCount": max((int(batch) // plan["rankMb"]) // 8, 1),
                "gpuCount": int(batch) // plan["rankMb"],
                "gradientAccumulation": 1,
            }
            for batch in ("64", "128", "256")
        } | {"rankMicrobatchSequences": plan["rankMb"]}
        report["poolPlan"] = {
            "poolTokens": 3_000_000_000,
            "baseTokens": 1_000_000_000,
            "extensionTokens": 2_000_000_000,
            "extensionManifest": f"{POOL_ROOT}/manifests/dclm_0802_nested_extension_1b_to_3b.json",
            "fullPoolManifest": f"{POOL_ROOT}/manifests/dclm_0802_nested_train_3b.json",
            "fixedLearningRateByBatch": plan["lr"],
            "initialWeightDecayByBatch": plan["initialWd"],
            "epochOneSourceWeightDecayByBatch": {
                batch: candidates[0] for batch, candidates in plan["initialWd"].items()
            },
            "rankMicrobatchSequences": plan["rankMb"],
            "gradientAccumulation": 1,
            "gpuTopologyByBatch": {
                batch: {
                    "gpuCountPerNode": min(int(batch) // plan["rankMb"], 8),
                    "nodeCount": max((int(batch) // plan["rankMb"]) // 8, 1),
                    "totalGpuCount": int(batch) // plan["rankMb"],
                }
                for batch in ("64", "128", "256")
            },
            "epochOneMode": "restore-predecay-model-optimizer-progress-reset-loader-on-new-2b",
            "laterEpochMode": "exact-coordinate-resume-reset-loader-and-reshuffle-full-3b",
        }
        destination = DATA_DIR / f"wsd_batch_size_{model}_pool3b.json"
        write_json_and_js(destination, report)

        html = source_html.read_text()
        html = html.replace(source["title"], report["title"])
        html = html.replace(
            f"data/wsd_batch_size_{model}.js",
            f"data/wsd_batch_size_{model}_pool3b.js",
        )
        batch_headers = "".join(f"<th>BS {batch}</th>" for batch in (64, 128, 256))
        for first_header, body_id in (
            ("Epoch", "validation-summary"),
            ("Optimizer steps", "optimizer-step-summary"),
            ("Epoch", "coordinate-summary"),
        ):
            html = re.sub(
                rf'(<thead><tr><th>{re.escape(first_header)}</th>)'
                rf'(?:<th>BS \d+</th>)+'
                rf'(</tr></thead><tbody id="{body_id}">)',
                rf'\1{batch_headers}\2',
                html,
            )
        (REPORT_DIR / f"wsd_batch_size_{model}_pool3b.html").write_text(html)


if __name__ == "__main__":
    main()
