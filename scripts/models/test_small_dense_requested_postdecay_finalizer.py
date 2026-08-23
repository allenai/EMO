#!/usr/bin/env python3
"""Focused tests for the user-requested small-model POST finalizers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_small_dense_requested_postdecay_finalizer as finalizer
import transition_small_dense_requested_postdecay_finalizers as transition


def result(epoch: int, value: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "validationExact": value,
        "checkpoint": f"/checkpoint/e{epoch}",
    }


class FinalizerTest(unittest.TestCase):
    def test_saturation_requires_three_post_results(self) -> None:
        self.assertFalse(finalizer.saturation_at({40: result(40, 3.2)}, 40))
        self.assertFalse(
            finalizer.saturation_at(
                {40: result(40, 3.2), 48: result(48, 3.1)}, 48
            )
        )
        self.assertTrue(
            finalizer.saturation_at(
                {
                    40: result(40, 3.2),
                    48: result(48, 3.1),
                    56: result(56, 3.1),
                },
                56,
            )
        )

    def test_explicit_stop_ignores_later_unrequested_results(self) -> None:
        results = {
            48: result(48, 3.2),
            56: result(56, 3.1),
            64: result(64, 3.0),
            72: result(72, 2.9),
        }
        value = finalizer.decision(
            {"lockedWd": "0.333"},
            [48],
            results,
            saturated=False,
            preserve_existing_selection=True,
        )
        self.assertEqual(value["evaluatedThroughEpoch"], 48)
        self.assertEqual(value["postDecaySourceEpochs"], [48])

    def test_requested_plan_matches_user_boundaries(self) -> None:
        self.assertEqual(transition.PLAN[("153m", 512)]["awaitEpoch"], 128)
        self.assertEqual(transition.PLAN[("153m", 256)]["epochs"], [64, 80, 96])
        self.assertEqual(
            transition.PLAN[("153m", 128)]["epochs"], [80, 96, 112, 128]
        )
        self.assertEqual(transition.PLAN[("474m", 256)]["epochs"], [48])
        self.assertEqual(
            transition.PLAN[("474m", 128)]["epochs"], [64, 72, 80, 88]
        )
        self.assertEqual(
            transition.PLAN[("474m", 64)]["epochs"],
            [40, 48, 56, 64, 72, 80, 88],
        )

    def test_waits_for_exact_current_stage_result(self) -> None:
        record = {"postDecayResults": {"128": {"status": "complete"}}}
        plan = transition.PLAN[("153m", 512)]
        self.assertTrue(transition.result_complete(record, plan))
        self.assertFalse(transition.result_complete({"postDecayResults": {}}, plan))

    def test_no_work_request_closes_from_existing_post_result(self) -> None:
        record = {
            "batchSequences": 512,
            "lockedWd": "0.3",
            "experiment": "01M0P7ZEJW37HSE45GCVKHBFEA",
            "recoveryOf": "01M0M1A7YTSE5KBQ73PSV4BEGQ",
            "requestedPostDecayFinalization": {
                "oldExperiment": "01M0M1A7YTSE5KBQ73PSV4BEGQ",
            },
            "postDecayResults": {
                "96": {**result(96, 3.341), "status": "complete"},
                "112": {**result(112, 3.337), "status": "complete"},
                "128": {**result(128, 3.334), "status": "complete"},
            },
        }
        transition.complete_without_gpu(record, transition.PLAN[("153m", 512)])
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["experiment"], "01M0M1A7YTSE5KBQ73PSV4BEGQ")
        self.assertEqual(record["postDecaySelection"]["status"], "stopped_by_user")
        self.assertFalse(record["postDecaySelection"]["postDecaySaturated"])
        self.assertEqual(
            record["postDecaySelection"]["postDecaySourceEpochs"], [96, 112, 128]
        )
        self.assertEqual(
            record["manualNoWorkFinalization"]["canceledPlaceholderExperiment"],
            "01M0P7ZEJW37HSE45GCVKHBFEA",
        )


if __name__ == "__main__":
    unittest.main()
