#!/usr/bin/env python3
"""Submit the two reviewer-requested 1.5B Pool-1B trajectories."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_dense_1b_repack_untied_embedwd_reviewer as runner
import submit_dense_1b_dr_wt_embedwd_grid as base

WORKSPACE = "ai2/flex2"
BASE_EXPERIMENT = "01KZD997FY92SE7CQM504CVDJW"
REGISTRY = Path(
    "reports/0802/data/wsd_1b_repack_untied_embedwd_reviewer.json"
)
REGISTRY_JS = REGISTRY.with_suffix(".js")
COORDINATES = (
    {
        "id": "dense-1b-dclm1b-bs64-lr1e-3-wd0.3-repack-untied-embedwd",
        "batchSequences": 64,
        "learningRate": "1e-3",
        "weightDecay": "0.3",
        "manifest": "scripts/models/manifests/dense-1b-repack-untied-embedwd-reviewer-bs64.json",
        "output": f"{runner.OUTPUT_ROOT}/bs64_lr1e-3_wd0.3",
    },
    {
        "id": "dense-1b-dclm1b-bs512-lr1e-3-wd1.0-repack-untied-embedwd",
        "batchSequences": 512,
        "learningRate": "1e-3",
        "weightDecay": "1.0",
        "manifest": "scripts/models/manifests/dense-1b-repack-untied-embedwd-reviewer-bs512.json",
        "output": f"{runner.OUTPUT_ROOT}/bs512_lr1e-3_wd1.0",
    },
)


def build_spec(item: dict[str, Any], revision: str, priority: str) -> dict[str, Any]:
    spec = base.spec_for(
        base_experiment=BASE_EXPERIMENT,
        manifest=str(item["manifest"]),
        lr=str(item["learningRate"]),
        wd=str(item["weightDecay"]),
        revision=revision,
        priority=priority,
        finalize_only=False,
        description=(
            f"Reviewer-requested 1.5B Pool-1B BS{item['batchSequences']} REPACK, "
            "untied embeddings, positive embedding WD; save every epoch and POST "
            "evaluate E8,E12,... through E32 or saturation."
        ),
    )
    task = spec["tasks"][0]
    task["arguments"][1] = (
        "scripts/models/run_dense_1b_repack_untied_embedwd_reviewer.py"
    )
    task["context"] = {
        "priority": priority,
        "minRuntime": "8h",
        "autoResume": True,
    }
    spec["retry"] = {"allowedTaskRetries": 8}
    return spec


def experiment_name(item: dict[str, Any]) -> str:
    return (
        f"dense-1b-dclm1b-bs{item['batchSequences']}-repack-untied-embedwd-"
        f"lr{item['learningRate']}-wd{item['weightDecay']}-reviewer-v1"
    )


def create(item: dict[str, Any], revision: str, priority: str) -> str:
    name = experiment_name(item)
    existing = base.existing_named_experiment(name)
    if existing:
        return existing
    output = base.run(
        ["beaker", "experiment", "create", "-", "--name", name, "--workspace", WORKSPACE],
        input_text=json.dumps(build_spec(item, revision, priority)),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError(f"submission returned no experiment ID for {name}")
    return identifiers[0]


def write_registry(created: list[tuple[dict[str, Any], str]], revision: str) -> None:
    value = {
        "policy": runner.POLICY,
        "updatedAt": datetime.now(tz=UTC).isoformat(),
        "model": "1.5B (repository Dense-1B / 1.279B parameters)",
        "pool": "dclm1b",
        "recipe": {
            "dynamicRepacking": True,
            "weightTying": False,
            "decayEmbeddings": True,
        },
        "checkpointEpochs": list(range(1, 33)),
        "evaluationEpochs": list(range(8, 33, 4)),
        "hardTerminalEpoch": 32,
        "stopCriterion": "first adjacent POST validationExact non-improvement",
        "trajectories": [
            {
                **item,
                "gpuCount": 8,
                "nodeCount": 1,
                "rankMicrobatchSequences": 8,
                "gradientAccumulation": int(item["batchSequences"]) // 64,
                "minRuntime": "8h",
                "status": "submitted",
                "experiment": experiment,
                "revision": revision,
                "postDecayResults": {},
            }
            for item, experiment in created
        ],
    }
    rendered = json.dumps(value, indent=2) + "\n"
    REGISTRY.write_text(rendered)
    REGISTRY_JS.write_text(
        "window.ICSL_1B_REPACK_UNTIED_EMBEDWD_REVIEWER="
        + rendered.rstrip()
        + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    base.validate_revision(args.revision)
    created: list[tuple[dict[str, Any], str]] = []
    for item in COORDINATES:
        manifest = json.loads(Path(str(item["manifest"])).read_text())
        runner.validate_config(manifest)
        if args.print_specs:
            print(json.dumps(build_spec(item, args.revision, args.priority), indent=2))
        elif args.submit:
            experiment = create(item, args.revision, args.priority)
            created.append((item, experiment))
            print(f"{item['id']}: {experiment}")
    if created:
        write_registry(created, args.revision)


if __name__ == "__main__":
    main()
