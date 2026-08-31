#!/usr/bin/env python3
"""Focused tests for allocated-slot evaluator runtime estimates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluator_min_runtime as runtime


class RuntimeEstimateTest(unittest.TestCase):
    def test_current_1b_e8_e16_evaluators_reserve_six_hours(self) -> None:
        for batch in (64, 128):
            estimate = runtime.estimate_min_runtime(
                model="1b",
                pool_tokens=3_000_000_000,
                batch_sequences=batch,
                epochs=[8, 16],
            )
            self.assertEqual(estimate["minRuntime"], "6h")
            self.assertGreater(estimate["reservedSeconds"], estimate["estimatedSeconds"])

    def test_future_small_evaluators_use_model_specific_rates(self) -> None:
        evaluator_474m = runtime.estimate_min_runtime(
            model="474m",
            pool_tokens=1_000_000_000,
            batch_sequences=128,
            epochs=[16],
        )
        evaluator_153m = runtime.estimate_min_runtime(
            model="153m",
            pool_tokens=1_000_000_000,
            batch_sequences=128,
            epochs=[32],
        )
        self.assertEqual(evaluator_474m["minRuntime"], "1h")
        self.assertEqual(evaluator_153m["minRuntime"], "30m")

    def test_later_frontiers_scale_the_reservation(self) -> None:
        evaluator_474m = runtime.estimate_min_runtime(
            model="474m",
            pool_tokens=1_000_000_000,
            batch_sequences=256,
            epochs=[32],
        )
        evaluator_153m = runtime.estimate_min_runtime(
            model="153m",
            pool_tokens=1_000_000_000,
            batch_sequences=256,
            epochs=[64],
        )
        self.assertEqual(evaluator_474m["minRuntime"], "90m")
        self.assertEqual(evaluator_153m["minRuntime"], "1h")

    def test_pool3b_small_evaluators_reserve_the_longer_decay(self) -> None:
        evaluator_474m = runtime.estimate_min_runtime(
            model="474m",
            pool_tokens=3_000_000_000,
            batch_sequences=128,
            epochs=[16],
        )
        evaluator_153m = runtime.estimate_min_runtime(
            model="153m",
            pool_tokens=3_000_000_000,
            batch_sequences=128,
            epochs=[32],
        )
        self.assertEqual(evaluator_474m["minRuntime"], "2h")
        self.assertEqual(evaluator_153m["minRuntime"], "90m")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported evaluator model"):
            runtime.estimate_min_runtime(
                model="7b", pool_tokens=1, batch_sequences=1, epochs=[1]
            )
        with self.assertRaisesRegex(ValueError, "epochs must be unique"):
            runtime.estimate_min_runtime(
                model="1b", pool_tokens=1, batch_sequences=1, epochs=[1, 1]
            )


if __name__ == "__main__":
    unittest.main()
