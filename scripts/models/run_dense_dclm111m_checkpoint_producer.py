#!/usr/bin/env python3
"""Run one retry-safe DCLM-111M producer with isolated POST branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_dense_dclm333m_checkpoint_producer as base

POLICY = "dense_dclm111m_integrated_producer_eval_v1"
POOL_SLUG = "dclm111m"
POOL_DISPLAY_NAME = "DCLM-111M"
MARKER_PREFIX = "DENSE_DCLM111M"
STATE_DIR_NAME = ".dclm111m_integrated_producer_eval_v1"
POOL_TOKENS = 111_000_000
EXPECTED_DATASET_MANIFEST = Path(
    "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_111m.json"
)
EXPECTED_MATERIALIZED_PATH = (
    "../sewonm/icsl/data/dclm_0802_nested_111m_from_333m/"
    "dclm_0802_repeated_train_111m_uint32.npy"
)
EXPECTED_BASE_TOKENS = 333_000_000
EXPECTED_BASE_MANIFEST = Path(
    "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_333m.json"
)

MODEL_POLICIES: dict[str, dict[str, Any]] = {
    "1b": {
        "batches": (64,),
        "lr": "1e-3",
        "wds": ("1.0", "3.0"),
        "retained_checkpoint_epochs": tuple(range(2, 25, 2)),
        "evaluation_epochs": tuple(range(4, 25, 4)),
        "max_epoch": 24,
        "base_experiment": "01M0WWNWS66NRG21QSKB87BKE7",
    },
    "474m": {
        "batches": (64,),
        "lr": "2e-3",
        "wds": ("0.3", "1.0"),
        "retained_checkpoint_epochs": tuple(range(4, 33, 4)),
        "evaluation_epochs": tuple(range(8, 33, 8)),
        "max_epoch": 32,
        "base_experiment": "01KZ7307CK7ZZQ1XCJ2QQ08KD4",
    },
    "153m": {
        "batches": (64,),
        "lr": "2e-3",
        "wds": ("0.3", "1.0"),
        "retained_checkpoint_epochs": tuple(range(8, 129, 8)),
        "evaluation_epochs": tuple(range(16, 129, 16)),
        "max_epoch": 128,
        "base_experiment": "01KZ6Q4DJ8J994A6SQ39MEGTZ2",
    },
}

BS32_POLICIES: dict[str, dict[str, Any]] = {
    "1b": {"lr": "5e-4", "wds": ("1.0", "3.0")},
    "474m": {"lr": "1e-3", "wds": ("0.3", "1.0")},
    "153m": {"lr": "1e-3", "wds": ("0.3", "1.0")},
}

BS64_474M_WD1_CONTINUATION_ID = "dense-474m-dclm111m-bs64-lr2e-3-wd1.0"


def configure_base() -> None:
    base.POLICY = POLICY
    base.POOL_SLUG = POOL_SLUG
    base.POOL_DISPLAY_NAME = POOL_DISPLAY_NAME
    base.MARKER_PREFIX = MARKER_PREFIX
    base.STATE_DIR_NAME = STATE_DIR_NAME
    base.POOL_TOKENS = POOL_TOKENS
    base.EXPECTED_DATASET_MANIFEST = EXPECTED_DATASET_MANIFEST
    base.EXPECTED_MATERIALIZED_PATH = EXPECTED_MATERIALIZED_PATH
    base.EXPECTED_BASE_TOKENS = EXPECTED_BASE_TOKENS
    base.EXPECTED_BASE_MANIFEST = EXPECTED_BASE_MANIFEST
    base.EXPECTED_COORDINATE_COUNT = 12
    base.EXPECTED_MODEL_COORDINATE_COUNTS = {"1b": 4, "474m": 4, "153m": 4}
    base.MODEL_POLICIES = MODEL_POLICIES
    base.BS32_POLICIES = BS32_POLICIES
    base.BS64_474M_CONTINUATION_TARGETS = (96,)
    base.BS64_474M_CONTINUATION_COORDINATE_IDS = (
        BS64_474M_WD1_CONTINUATION_ID,
    )
    base.BS64_474M_CONTINUATION_RETAIN_INTERVAL = 4
    base.BS64_474M_CONTINUATION_EVAL_INTERVAL = 8
    base.BS64_153M_WD03_CONTINUATION_TARGETS = ()
    base.BS64_474M_LR1E3_WD03_PROBE = "unused-dclm111m-coordinate"
    base.ALL_CONTINUATION_TARGETS = (96,)


configure_base()

load_manifest = base.load_manifest
coordinate_for_target = base.coordinate_for_target
continuation_source_epoch = base.continuation_source_epoch
validate_coordinate = base.validate_coordinate
validate_dataset_manifest = base.validate_dataset_manifest
gpu_count = base.gpu_count
NPROC_PER_NODE = base.NPROC_PER_NODE
runtime_estimate = base.runtime_estimate
state_dir = base.state_dir
stable_step = base.stable_step
total_step = base.total_step
base_arguments = base.base_arguments
postdecay_arguments = base.postdecay_arguments
run = base.run


if __name__ == "__main__":
    base.main()
