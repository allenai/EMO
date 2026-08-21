from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_small_dense_locked_wd_predecay_chain as policy


def config() -> dict:
    return {
        "model": "474m",
        "globalSequences": 64,
        "nprocPerNode": 4,
        "rankMicrobatchSequences": 16,
        "warmupSteps": 384,
        "learningRate": "0.002",
        "wdLadder": ["0.01", "0.033", "0.1", "0.3", "1.0"],
        "baselineOptimalWd": "0.1",
        "initialWds": ["0.033", "0.1", "0.3"],
        "initialTargets": [1, 2, 4, 8, 16, 24],
        "epochIncrement": 8,
        "outputRoot": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm1b",
        "runSuffix": "test",
        "policy": policy.POLICY,
        "lockedWd": "0.1",
        "postDecaySourceCount": 3,
        "comparisonPolicy": "within_phase_only",
        "historicalPreDecayThroughEpoch": 3,
    }


def result(value: float) -> dict:
    return {"validationExact": value}


class LockedWdPreDecayPolicyTest(unittest.TestCase):
    def test_saturation_uses_predecay_sequence_and_requires_three_points(self) -> None:
        self.assertIsNone(policy.first_saturation({1: result(4.0), 2: result(4.1)}))
        self.assertEqual(
            policy.first_saturation(
                {1: result(4.0), 2: result(3.9), 3: result(3.9), 4: result(3.8)}
            ),
            3,
        )

    def test_scheduler_and_evaluation_phases_are_isolated(self) -> None:
        cfg = config()
        source = Path(policy.locked_output(cfg)) / "step1"
        constant = policy.constant_training_arguments(cfg, 2, source, "constant")
        self.assertTrue(any("ConstantScheduler" in value for value in constant))
        self.assertFalse(any("scheduler.WSD" in value for value in constant))
        post = policy.postdecay_training_arguments(
            cfg,
            2,
            source,
            Path("/weka/oe-training-default/sewonm/icsl/models/post"),
            "post",
        )
        self.assertTrue(any("scheduler.WSD" in value for value in post))
        evaluation = policy.evaluation_arguments(
            cfg,
            source,
            Path("/weka/oe-training-default/sewonm/icsl/models/eval"),
            "eval",
        )
        self.assertIn("--trainer.callbacks.downstream_evaluator.eval_on_startup=true", evaluation)
        self.assertIn("--trainer.callbacks.downstream_evaluator.eval_on_finish=false", evaluation)
        self.assertIn("--load_trainer_state=false", evaluation)

    def test_discovery_requires_complete_exact_predecay_lineage(self) -> None:
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(policy, "locked_output", return_value=root):
                for epoch in (1, 2, 3):
                    (root / f"step{policy.adaptive.stable_step(epoch, 64)}").mkdir()
                self.assertEqual(policy.discover_predecay_epochs(cfg), [1, 2, 3])
                (root / f"step{policy.adaptive.stable_step(2, 64)}").rmdir()
                with self.assertRaisesRegex(RuntimeError, "missing epochs"):
                    policy.discover_predecay_epochs(cfg)

    def test_future_predecay_checkpoints_use_separate_constant_output(self) -> None:
        cfg = config()
        self.assertEqual(policy.checkpoint_for_epoch(cfg, 3).parent, policy.locked_output(cfg))
        self.assertEqual(policy.checkpoint_for_epoch(cfg, 4).parent, policy.constant_output(cfg))

    def test_postdecay_selects_last_three_using_postdecay_scores(self) -> None:
        cfg = config()
        values = {4: 3.4, 5: 3.2, 6: 3.3}

        def fake_postdecay(_: dict, epoch: int) -> dict:
            return {
                "epoch": epoch,
                "phase": "post_decay",
                "comparisonGroup": "post_decay",
                "validationExact": values[epoch],
                "checkpoint": f"/post/e{epoch}",
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(policy, "policy_state_dir", return_value=root),
                mock.patch.object(policy, "run_postdecay", side_effect=fake_postdecay),
            ):
                policy.finish_postdecay(cfg, 6, [1, 2, 3, 4, 5, 6])
                selection = json.loads(policy.selection_path(cfg).read_text())
        self.assertEqual(selection["postDecaySourceEpochs"], [4, 5, 6])
        self.assertEqual(selection["selectedPostDecayEpoch"], 5)
        self.assertEqual(selection["preDecayDecisionGroup"], "pre_decay")
        self.assertEqual(selection["postDecaySelectionGroup"], "post_decay")


if __name__ == "__main__":
    unittest.main()
