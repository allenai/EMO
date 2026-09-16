#!/usr/bin/env python3
"""Focused checkpoint-policy tests for the 153M Pool-3B LR4e-3 chain."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dense_153m_pool3b_bs256_lr4e3_wd01_saturation as runner
import submit_dense_153m_pool3b_bs256_lr4e3_wd01_saturation as submitter


class RecoveryCheckpointPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.item = runner.load(runner.DEFAULT_MANIFEST)

    def test_recovery_interval_is_one_epoch_of_steps(self) -> None:
        self.assertEqual(
            runner.RECOVERY_SAVE_INTERVAL_STEPS,
            runner.checkpoint_step(1),
        )

    def test_training_uses_ephemeral_recovery_and_permanent_pd_steps(self) -> None:
        arguments = runner.train_arguments(
            self.config,
            self.item,
            runner.OUTPUT / f"step{runner.checkpoint_step(288)}",
            [320],
        )
        self.assertIn(
            "--trainer.callbacks.checkpointer.ephemeral_save_interval="
            f"{runner.RECOVERY_SAVE_INTERVAL_STEPS}",
            arguments,
        )
        self.assertIn(
            f"--trainer.callbacks.checkpointer.fixed_steps=[{runner.checkpoint_step(320)}]",
            arguments,
        )
        self.assertEqual(
            1,
            sum(
                argument.startswith(
                    "--trainer.callbacks.checkpointer.ephemeral_save_interval="
                )
                for argument in arguments
            ),
        )

    def test_replacement_registration_preserves_resolved_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            report_js = Path(temp_dir) / "report.js"
            report.write_text(
                '{"pool3bLearningRateProbes": ['
                '{"id": "dense-153m-dclm3b-bs256-lr4e-3-wd0.1", '
                '"experiment": "old", "jobs": ["old-job"], "revision": "old-rev", '
                '"resolvedCheckpointEpochs": [320], "resolvedPostEpochs": [320], '
                '"postDecayResults": {"320": {"validationExact": 3.2052}}}'
                ']}'
            )
            with (
                mock.patch.object(submitter, "REPORT", report),
                mock.patch.object(submitter, "REPORT_JS", report_js),
            ):
                submitter.register("new", "new-rev", replace_existing=True)
            registered = __import__("json").loads(report.read_text())[
                "pool3bLearningRateProbes"
            ][0]
            self.assertEqual(registered["experiment"], "new")
            self.assertEqual(registered["jobs"], [])
            self.assertNotIn("job", registered)
            self.assertEqual(registered["resolvedCheckpointEpochs"], [320])
            self.assertEqual(registered["resolvedPostEpochs"], [320])
            self.assertEqual(
                registered["postDecayResults"]["320"]["validationExact"],
                3.2052,
            )
            self.assertEqual(
                registered["experimentHistory"][-1]["status"],
                "canceled_to_enable_per_epoch_recovery_checkpointing",
            )

    def test_cleanup_candidates_exclude_all_permanent_checkpoints(self) -> None:
        candidates = set(runner.recovery_checkpoint_steps_through(320))
        self.assertNotIn(runner.checkpoint_step(1), candidates)
        for epoch in runner.CHECKPOINT_EPOCHS:
            if epoch <= 320:
                self.assertNotIn(runner.checkpoint_step(epoch), candidates)
        self.assertIn(runner.RECOVERY_SAVE_INTERVAL_STEPS * 2, candidates)
        self.assertLess(max(candidates), runner.checkpoint_step(320))

    def test_latest_complete_recovery_checkpoint_is_used_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            recovery_step = runner.RECOVERY_SAVE_INTERVAL_STEPS * 319
            complete_steps = {runner.checkpoint_step(288), recovery_step}

            def complete(path: Path, _: int) -> bool:
                return int(path.name.removeprefix("step")) in complete_steps

            with (
                mock.patch.object(runner, "OUTPUT", output),
                mock.patch.object(
                    runner.distributed,
                    "checkpoint_complete",
                    side_effect=complete,
                ),
            ):
                resume = runner.latest_resume_checkpoint(
                    output / f"step{runner.checkpoint_step(1)}",
                    320,
                )
            self.assertEqual(resume, output / f"step{recovery_step}")

    def test_post_decay_uses_latest_complete_recovery_checkpoint(self) -> None:
        epoch = 320
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            candidates = runner.post_recovery_checkpoint_steps(epoch)
            recovery_step = candidates[-2]

            def complete(path: Path, _: int) -> bool:
                return path.name == f"step{recovery_step}"

            with mock.patch.object(
                runner.distributed,
                "checkpoint_complete",
                side_effect=complete,
            ):
                resume = runner.latest_post_resume_checkpoint(
                    output,
                    runner.OUTPUT / f"step{runner.checkpoint_step(epoch)}",
                    epoch,
                )
            self.assertEqual(resume, output / f"step{recovery_step}")

    def test_post_decay_arguments_save_rolling_recovery_checkpoints(self) -> None:
        epoch = 320
        source = runner.OUTPUT / f"step{runner.checkpoint_step(epoch)}"
        post_output = runner.STATE_DIR / "post_decay_runs" / f"e{epoch}"
        arguments = runner.evaluator.postdecay_arguments(
            self.config,
            self.item,
            epoch,
            source,
            post_output,
            "test-post",
        )
        arguments = runner.producer.common.upsert(
            arguments,
            "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
            (
                "--trainer.callbacks.checkpointer.ephemeral_save_interval="
                f"{runner.RECOVERY_SAVE_INTERVAL_STEPS}"
            ),
        )
        self.assertIn(
            "--trainer.callbacks.checkpointer.ephemeral_save_interval="
            f"{runner.RECOVERY_SAVE_INTERVAL_STEPS}",
            arguments,
        )


if __name__ == "__main__":
    unittest.main()
