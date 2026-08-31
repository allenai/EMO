#!/usr/bin/env python3
"""Focused provenance tests for standalone Pool-3B small evaluators."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dense_small_pool3b_checkpoint_evaluator as evaluator

MANIFEST = SCRIPT_DIR / "manifests" / "dense-small-pool3b-checkpoint-producers-v2.json"


class ProducerOutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(MANIFEST.read_text())
        cls.item = next(
            item
            for item in config["producerCoordinates"]
            if item["id"] == "dense-153m-dclm3b-bs256-lr2e-3-wd0.033"
        )

    def test_canonical_output_is_accepted(self) -> None:
        output = evaluator.producer_output(self.item, str(self.item["output"]))
        self.assertEqual(output, Path(self.item["output"]))

    def test_isolated_recovery_output_is_accepted_and_used(self) -> None:
        output = str(self.item["output"]) + "_throughput_recovery_r3"
        source = evaluator.source_checkpoint(self.item, 32, output)
        self.assertEqual(source.parent, Path(output))
        self.assertEqual(source.name, "step82397")

    def test_unrelated_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unauthorized evaluator producer output"):
            evaluator.producer_output(self.item, "/weka/oe-training-default/wrong")


if __name__ == "__main__":
    unittest.main()
