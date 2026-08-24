#!/usr/bin/env python3
"""Focused tests for reopened small-model POST-only continuations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import resume_small_dense_user_stopped_postdecay_chains as submitter
import run_small_dense_postdecay_only_continuation as runner


def result(epoch: int, validation: float, batch: int, wd: str) -> dict[str, object]:
    step = runner.adaptive.total_step(epoch, batch)
    source_step = runner.adaptive.stable_step(epoch, batch)
    root = "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm1b"
    return {
        "epoch": epoch,
        "status": "complete",
        "phase": "post_decay",
        "comparisonGroup": "post_decay",
        "validation": round(validation, 3),
        "validationExact": validation,
        "checkpoint": f"{root}/post/e{epoch}/step{step}",
        "sourcePreDecayCheckpoint": f"{root}/constant/step{source_step}",
        "wd": wd,
    }


def record(batch: int, boundary: int, wd: str) -> dict[str, object]:
    epochs = [boundary - 32, boundary - 16, boundary]
    results = {
        str(epoch): result(epoch, 3.4 - index * 0.01, batch, wd)
        for index, epoch in enumerate(epochs)
    }
    return {
        "batchSequences": batch,
        "lockedWd": wd,
        "postDecaySaturated": False,
        "postDecaySelection": {"status": "stopped_by_user"},
        "postDecayResults": results,
        "rankMicrobatchSequences": 16,
        "gradientAccumulation": batch // 128,
        "lr": "2e-3",
        "wdLadder": ["0.01", "0.033", "0.1", "0.3", "1.0"],
        "initialTargets": [1, 2, 4, 8, 16, 32, 48],
        "epochIncrement": 16,
    }


class PostOnlyContinuationTest(unittest.TestCase):
    def test_exact_user_boundaries(self) -> None:
        self.assertEqual(submitter.TARGETS, {128: 160, 256: 144, 512: 128})

    def test_manifest_uses_exact_boundary_and_isolated_output(self) -> None:
        value = submitter.manifest_from_record(record(256, 144, "0.1"), 144)
        self.assertEqual(value["resumeEpoch"], 144)
        self.assertTrue(value["resumeCheckpoint"].endswith("/step123596"))
        self.assertIn("postdecay_only_continuation_v1", value["continuationRoot"])
        self.assertFalse(value["preDecayEvaluation"])
        runner.validate_config(value)

    def test_saturation_is_sequential_post_only(self) -> None:
        self.assertFalse(
            runner.saturated(
                {"validationExact": 3.31}, {"validationExact": 3.30}
            )
        )
        self.assertTrue(
            runner.saturated(
                {"validationExact": 3.30}, {"validationExact": 3.30}
            )
        )
        self.assertTrue(
            runner.saturated(
                {"validationExact": 3.30}, {"validationExact": 3.31}
            )
        )

    def test_decision_uses_latest_three_post_results(self) -> None:
        config = submitter.manifest_from_record(record(128, 160, "0.1"), 160)
        results = runner.prior_results(config)
        results[176] = result(176, 3.39, 128, "0.1")
        value = runner.decision(config, results, 176)
        self.assertEqual(value["postDecaySourceEpochs"], [144, 160, 176])
        self.assertTrue(value["postDecaySaturated"])
        self.assertEqual(value["selectedPostDecayEpoch"], 160)

    def test_constant_training_disables_frontier_evaluation(self) -> None:
        config = submitter.manifest_from_record(record(512, 128, "0.3"), 128)
        arguments = runner.constant_training_arguments(
            config,
            144,
            Path(config["resumeCheckpoint"]),
            "test-run",
        )
        self.assertIn(
            "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
            arguments,
        )
        heldout = next(
            value
            for value in arguments
            if value.startswith("--trainer.callbacks.heldout_evaluator=")
        )
        self.assertNotIn("eval_on_finish: true", heldout)


if __name__ == "__main__":
    unittest.main()
