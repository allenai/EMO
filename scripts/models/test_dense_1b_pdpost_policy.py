#!/usr/bin/env python3
"""Focused tests for the guarded Dense-1B PD/POST controller."""

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
    def test_pd_requires_three_and_then_uses_only_latest_pair(self) -> None:
        self.assertIsNone(runner.latest_predecay_saturation({1: result(4.0), 2: result(3.0)}))
        self.assertIsNone(
            runner.latest_predecay_saturation(
                {1: result(4.0), 2: result(3.0), 4: result(2.9)}
            )
        )
        self.assertEqual(
            runner.latest_predecay_saturation(
                {1: result(4.0), 2: result(3.0), 4: result(3.0)}
            ),
            4,
        )

    def test_post_improvement_overrides_provisional_pd_stop(self) -> None:
        self.assertFalse(
            runner.postdecay_saturated(
                {1: result(3.2), 2: result(3.1), 4: result(3.0)}
            )
        )
        self.assertTrue(
            runner.postdecay_saturated(
                {1: result(3.2), 2: result(3.0), 4: result(3.0)}
            )
        )

    def test_frontier_ladder_is_legacy_dense_1b_ladder(self) -> None:
        config = {"initialTargets": [1, 2, 4], "epochIncrement": 4}
        self.assertEqual([runner.target_at(config, i) for i in range(7)], [1, 2, 4, 8, 12, 16, 20])


class PruningTest(unittest.TestCase):
    @staticmethod
    def coordinate(wd: str, values: list[float]) -> dict:
        epochs = [1, 2, 4][: len(values)]
        return {
            "id": f"drwtembwd128-lr1e-3-wd{wd}",
            "policy": runner.POLICY,
            "variant": "DR+WT+EmbedWD",
            "batchSequences": 128,
            "lr": "1e-3",
            "wd": wd,
            "status": "running",
            "activePhase": "pre_decay_train",
            "preDecayResults": {
                str(epoch): {"validationExact": value}
                for epoch, value in zip(epochs, values, strict=True)
            },
        }

    def test_higher_wd_win_waits_for_three_lower_pd_sources(self) -> None:
        report = {"runs": [self.coordinate("0.3", [3.2, 3.1]), self.coordinate("1.0", [3.3, 3.0])]}
        self.assertEqual(monitor.wd_prune_requests(report), [])
        self.assertEqual(
            report["runs"][0]["pruneDeferredReason"],
            "fewer_than_three_completed_pd_sources",
        )

    def test_higher_wd_strict_win_requests_lower_wd_finalizer(self) -> None:
        low = self.coordinate("0.3", [3.2, 3.1, 3.0])
        high = self.coordinate("1.0", [3.3, 3.0, 2.9])
        requested = monitor.wd_prune_requests({"runs": [low, high]})
        self.assertEqual([record["id"] for record in requested], [low["id"]])
        self.assertEqual(low["pruneRequested"]["epoch"], 2)

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
                                {"batchSequences": 32, "experiment": f"{model}-orig", "status": "complete"}
                            ],
                            "adaptiveDrWtEmbedWdChains": [
                                {
                                    "batchSequences": batch,
                                    "policy": submitter.SMALL_POLICY,
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


if __name__ == "__main__":
    unittest.main()
