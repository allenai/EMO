#!/usr/bin/env python3
"""Focused tests for the guarded Dense-1B all-POST controller."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import monitor_dense_1b_dr_wt_embedwd_grid as monitor
import run_dense_1b_dr_wt_embedwd_grid as runner
import submit_dense_1b_dr_wt_embedwd_grid as submitter


def result(value: float) -> dict[str, float]:
    return {"validationExact": value}


class SaturationTest(unittest.TestCase):
    def test_post_requires_three_and_then_uses_only_latest_pair(self) -> None:
        self.assertFalse(runner.postdecay_saturated({8: result(3.2), 12: result(3.1)}))
        self.assertFalse(
            runner.postdecay_saturated({8: result(3.2), 12: result(3.1), 16: result(3.0)})
        )
        self.assertTrue(
            runner.postdecay_saturated({8: result(3.2), 12: result(3.0), 16: result(3.0)})
        )

    def test_frontier_ladder_is_legacy_dense_1b_ladder(self) -> None:
        config = {"initialTargets": [1, 2, 4], "epochIncrement": 4}
        self.assertEqual([runner.target_at(config, i) for i in range(7)], [1, 2, 4, 8, 12, 16, 20])

    def test_post_starts_at_e8_and_early_frontiers_are_checkpoint_only(self) -> None:
        self.assertEqual(runner.POST_DECAY_START_EPOCH, 8)
        self.assertEqual(runner.CHECKPOINT_ONLY_EPOCHS, [1, 2, 4])
        with self.assertRaisesRegex(ValueError, "disabled for E1/E2/E4"):
            runner.run_postdecay({}, "1e-3", "0.3", 4)

    def test_terminal_selection_uses_latest_three_e8_or_later_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "globalSequences": 128,
                "variant": "DR+WT+EmbedWD",
                "initialTargets": [1, 2, 4],
                "epochIncrement": 4,
                "coordinates": [{"lr": "1e-3", "wd": "0.3", "output": directory}],
            }
            values = {8: 3.2, 12: 3.0, 16: 3.0}

            results = {
                epoch: {
                    "epoch": epoch,
                    "validationExact": value,
                    "checkpoint": f"/post/e{epoch}",
                }
                for epoch, value in values.items()
            }
            with (
                mock.patch.object(
                    runner,
                    "run_postdecay",
                    side_effect=lambda _config, _lr, _wd, epoch: results[epoch],
                ),
                mock.patch.object(runner, "recover_postdecay_results", return_value=results),
            ):
                self.assertTrue(
                    runner.finish_postdecay(
                        config,
                        "1e-3",
                        "0.3",
                        16,
                        [1, 2, 4, 8, 12, 16],
                        pruned=False,
                    )
                )
            selection = json.loads(runner.selection_path(config, "1e-3", "0.3").read_text())
            self.assertEqual(selection["postDecayFinalizerSourceEpochs"], [8, 12, 16])
            self.assertEqual(selection["postDecayEvaluatedEpochs"], [8, 12, 16])
            self.assertNotIn("preDecayDecisionGroup", selection)


class PruningTest(unittest.TestCase):
    @staticmethod
    def coordinate(wd: str, values: list[float]) -> dict:
        epochs = [8, 12, 16][: len(values)]
        return {
            "id": f"drwtembwd128-lr1e-3-wd{wd}",
            "policy": runner.POLICY,
            "variant": "DR+WT+EmbedWD",
            "batchSequences": 128,
            "lr": "1e-3",
            "wd": wd,
            "status": "running",
            "activePhase": "pre_decay_train",
            "postDecayResults": {
                str(epoch): {"validationExact": value}
                for epoch, value in zip(epochs, values, strict=True)
            },
        }

    def test_higher_wd_win_waits_for_three_lower_post_sources(self) -> None:
        report = {"runs": [self.coordinate("0.3", [3.2, 3.1]), self.coordinate("1.0", [3.3, 3.0])]}
        self.assertEqual(monitor.wd_prune_requests(report), [])
        self.assertEqual(
            report["runs"][0]["pruneDeferredReason"],
            "fewer_than_three_completed_post_sources",
        )

    def test_higher_wd_strict_win_requests_lower_wd_finalizer(self) -> None:
        low = self.coordinate("0.3", [3.2, 3.1, 3.0])
        high = self.coordinate("1.0", [3.3, 3.0, 2.9])
        requested = monitor.wd_prune_requests({"runs": [low, high]})
        self.assertEqual([record["id"] for record in requested], [low["id"]])
        self.assertEqual(low["pruneRequested"]["epoch"], 12)
        self.assertEqual(low["pruneRequested"]["comparisonGroup"], "post_decay")

    def test_equal_high_wd_result_does_not_prune(self) -> None:
        low = self.coordinate("0.3", [3.2, 3.1, 3.0])
        high = self.coordinate("1.0", [3.3, 3.1, 3.0])
        self.assertEqual(monitor.wd_prune_requests({"runs": [low, high]}), [])

    def test_partial_submission_cannot_make_batch_chain_complete(self) -> None:
        chain = {
            "id": "dense-1b-bs128-dr-wt-embwd-grid",
            "batchSequences": 128,
            "experimentsByCoordinate": {"low": "01LOW"},
        }
        report = {
            "runs": [
                {
                    "id": "low",
                    "method": "drwtembwd128",
                    "policy": runner.POLICY,
                    "batchSequences": 128,
                    "status": "complete",
                    "postDecaySelection": {"status": "complete"},
                    "selectedPostDecayEpoch": 4,
                    "selectedPostDecayValidationExact": 3.0,
                    "lr": "1e-3",
                    "wd": "0.3",
                    "variant": "DR+WT+EmbedWD",
                },
                {
                    "id": "high",
                    "method": "drwtembwd128",
                    "policy": runner.POLICY,
                    "batchSequences": 128,
                    "status": "held",
                    "lr": "1e-3",
                    "wd": "1.0",
                    "variant": "DR+WT+EmbedWD",
                },
            ]
        }
        monitor.aggregate_chain(report, chain)
        self.assertNotEqual(chain.get("status"), "complete")


class GateTest(unittest.TestCase):
    def test_plan_upsert_preserves_live_phase_state(self) -> None:
        records = [
            {
                "id": "coordinate",
                "status": "running",
                "experiment": "01EXPERIMENT",
                "preDecayResults": {"4": {"validationExact": 3.0}},
                "postDecayResults": {"2": {"validationExact": 3.1}},
                "activePhase": "pre_decay_train",
            }
        ]
        updated = submitter.upsert(
            records,
            "coordinate",
            {
                "id": "coordinate",
                "status": "planned",
                "preDecayResults": {},
                "postDecayResults": {},
            },
        )
        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["preDecayResults"]["4"]["validationExact"], 3.0)
        self.assertEqual(updated["postDecayResults"]["2"]["validationExact"], 3.1)
        self.assertEqual(updated["activePhase"], "pre_decay_train")

    def test_submit_gate_is_idempotent_for_existing_coordinate_experiments(self) -> None:
        runs = []
        chains = []
        for batch, config in submitter.PRIMARY.items():
            chains.append(
                {
                    "id": submitter.chain_id(batch),
                    "status": "complete",
                    "experimentsByCoordinate": {},
                }
            )
            for lr, wd in config["coordinates"]:
                runs.append(
                    {
                        "id": submitter.run_id(batch, lr, wd),
                        "experiment": f"existing-{batch}-{lr}-{wd}",
                        "status": "complete",
                    }
                )
        report = {"runs": runs, "drWtEmbedWdGridChains": chains}
        with mock.patch.object(
            submitter, "create_experiment", side_effect=AssertionError("must not resubmit")
        ):
            submitter.release_primary(report, "a" * 40, "urgent", 10)
        self.assertTrue(all(record["status"] == "complete" for record in runs))
        self.assertTrue(all(chain["status"] == "complete" for chain in chains))

    def test_small_gate_requires_confirmed_post_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for model in ("474m", "153m"):
                path = Path(directory) / f"wsd_batch_size_{model}.json"
                path.write_text(
                    json.dumps(
                        {
                            "batchSweeps": [
                                {
                                    "batchSequences": 32,
                                    "experiment": f"{model}-orig",
                                    "status": "complete",
                                }
                            ],
                            "adaptiveDrWtEmbedWdChains": [
                                {
                                    "batchSequences": batch,
                                    "policy": "locked_wd_predecay_saturation_v1",
                                    "status": "complete",
                                    "postDecaySelection": {
                                        "status": "complete",
                                        "postDecaySaturated": batch != 512 or model == "474m",
                                    },
                                }
                                for batch in (64, 128, 256, 512)
                            ],
                        }
                    )
                )
                paths.append(path)
            with mock.patch.object(submitter, "SMALL_REPORTS", tuple(paths)):
                count, _ = submitter.successful_small_chains()
            self.assertEqual(count, 9)

    def test_small_gate_accepts_saturated_requested_finalizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for model in ("474m", "153m"):
                path = Path(directory) / f"wsd_batch_size_{model}.json"
                path.write_text(
                    json.dumps(
                        {
                            "batchSweeps": [
                                {
                                    "batchSequences": 32,
                                    "experiment": f"{model}-orig",
                                    "status": "complete",
                                }
                            ],
                            "adaptiveDrWtEmbedWdChains": [
                                {
                                    "batchSequences": batch,
                                    "policy": (
                                        "locked_wd_requested_postdecay_finalizer_v1"
                                        if batch == 64
                                        else "locked_wd_predecay_saturation_v1"
                                    ),
                                    "status": "complete",
                                    "postDecaySelection": {
                                        "status": "complete",
                                        "postDecaySaturated": True,
                                    },
                                }
                                for batch in (64, 128, 256, 512)
                            ],
                        }
                    )
                )
                paths.append(path)
            with mock.patch.object(submitter, "SMALL_REPORTS", tuple(paths)):
                count, _ = submitter.successful_small_chains()
            self.assertEqual(count, 10)


if __name__ == "__main__":
    unittest.main()
