from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_1b_independent_pool_table5 as runner  # noqa: E402
import submit_dense_1b_independent_pool_table5 as submitter  # noqa: E402


PLAN = SCRIPTS / "manifests" / "dense-1b-independent-pools-table5-v1.json"


def load_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_plan_is_exact_two_by_four_table5_replication() -> None:
    plan = load_plan()
    runner.validate_plan(plan)
    assert len(plan["trajectories"]) == 8
    assert {(item["pool"], item["batchSequences"]) for item in plan["trajectories"]} == {
        (pool, batch) for pool in ("b", "c") for batch in (32, 64, 128, 256)
    }
    assert len({item["output"] for item in plan["trajectories"]}) == 8


def test_training_arguments_are_original_shuffled_and_checkpoint_every_epoch() -> None:
    plan = load_plan()
    for item in plan["trajectories"]:
        runner.configure_base(item)
        config = runner.base_config(item)
        arguments = runner.base.predecay_training_arguments(
            config,
            str(item["lr"]),
            str(item["wd"]),
            int(item["targetEpoch"]),
            None,
            "argument-audit",
            None,
        )
        assert f"--dataset.subset_manifest={item['dataManifest']}" in arguments
        assert "--model.tie_embeddings=false" in arguments
        assert "--decay-embeddings" not in arguments
        assert "--dynamic-repacking" not in arguments
        assert "--fixed-data-order" not in arguments
        assert f"--data_loader.global_batch_size={item['batchSequences'] * 4096}" in arguments
        assert "--train_module.rank_microbatch_size=16384" in arguments
        fixed = next(
            argument
            for argument in arguments
            if argument.startswith("--trainer.callbacks.checkpointer.fixed_steps=")
        )
        assert fixed.count(",") + 1 == item["targetEpoch"]


def test_submission_is_one_unallocated_eight_gpu_node(monkeypatch) -> None:
    def fake_spec_for(**_: object) -> dict:
        return {
            "tasks": [
                {
                    "arguments": [],
                    "resources": {},
                    "context": {"minRuntime": "8h"},
                    "envVars": [{"name": "GIT_REF", "value": "0" * 40}],
                }
            ]
        }

    monkeypatch.setattr(submitter.beaker_base, "spec_for", fake_spec_for)
    for item in load_plan()["trajectories"]:
        spec = submitter.build_spec(
            item,
            revision="0" * 40,
            priority="urgent",
            base_experiment=submitter.BASE_EXPERIMENT,
        )
        task = spec["tasks"][0]
        assert task["resources"]["gpuCount"] == 8
        assert task["context"] == {"priority": "urgent", "autoResume": True}
        assert "minRuntime" not in task["context"]
        assert "replicas" not in task
        assert spec["retry"] == {"allowedTaskRetries": 8}
