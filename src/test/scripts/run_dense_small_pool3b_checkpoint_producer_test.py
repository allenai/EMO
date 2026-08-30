from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_small_pool3b_checkpoint_producer as producer


MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "models"
    / "manifests"
    / "dense-small-pool3b-checkpoint-producers-v2.json"
)


def argument(arguments: list[str], prefix: str) -> str:
    values = [value for value in arguments if value.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_manifest_has_exact_authorized_coordinate_and_directory_semantics() -> None:
    config = producer.load_manifest(MANIFEST)
    assert len(config["producerCoordinates"]) == 8
    for item in config["producerCoordinates"]:
        producer.validate_coordinate(config, item, check_filesystem=False)
        assert Path(item["sourceCheckpoint"]) == producer.expected_source(item)
        assert Path(item["output"]) == producer.expected_output(item)
        assert "_dclm1b/" in item["sourceCheckpoint"]
        assert "_dclm3b/" in item["output"]
        assert "_dr_wt_embwd_" in item["output"]


def test_bridge_uses_only_fresh_extension_and_reaches_pool3b_predecay_e1() -> None:
    config = producer.load_manifest(MANIFEST)
    for item in config["producerCoordinates"]:
        batch = int(item["batchSequences"])
        arguments = producer.bridge_arguments(config, item)
        source_step = producer.stable_step(
            1, producer.SOURCE_POOL_TOKENS, batch
        )
        target_step = producer.stable_step(
            1, producer.TARGET_POOL_TOKENS, batch
        )
        assert Path(item["sourceCheckpoint"]).name == f"step{source_step}"
        assert target_step - source_step == (3434 if batch == 128 else 1717)
        assert argument(arguments, "--dataset.subset_manifest=") == (
            f"--dataset.subset_manifest={config['bridgeManifest']}"
        )
        assert argument(arguments, "--trainer.max_duration=") == (
            f"--trainer.max_duration={{value: {target_step}, unit: steps}}"
        )
        assert argument(arguments, "--trainer.load_path=") == (
            f"--trainer.load_path={item['sourceCheckpoint']}"
        )
        assert "--trainer.reset_data_loader_state_on_load_path=true" in arguments
        assert "--dynamic-repacking" not in arguments


def test_continuation_repeats_repacked_shuffled_pool3b_without_evaluation_or_decay() -> None:
    config = producer.load_manifest(MANIFEST)
    for item in config["producerCoordinates"]:
        arguments = producer.continuation_arguments(config, item)
        assert argument(arguments, "--dataset.subset_manifest=") == (
            f"--dataset.subset_manifest={config['repeatedManifest']}"
        )
        assert "--dynamic-repacking" in arguments
        assert "--trainer.reset_data_loader_state_on_load_path=true" in arguments
        assert argument(
            arguments, "--trainer.callbacks.downstream_evaluator.tasks="
        ) == "--trainer.callbacks.downstream_evaluator.tasks=[]"
        assert argument(
            arguments, "--trainer.callbacks.downstream_evaluator.eval_on_finish="
        ) == "--trainer.callbacks.downstream_evaluator.eval_on_finish=false"
        heldout = argument(arguments, "--trainer.callbacks.heldout_evaluator=")
        assert "eval_on_finish: false" in heldout
        assert "ConstantScheduler" in argument(arguments, "--train_module.scheduler=")
        assert not any("WSD" in value for value in arguments)


def test_checkpoint_ladders_end_at_e256() -> None:
    config = producer.load_manifest(MANIFEST)
    for item in config["producerCoordinates"]:
        epochs = producer.target_epochs(item, int(config["maxEpoch"]))
        increment = 16 if item["model"] == "474m" else 32
        assert epochs == list(range(increment, 257, increment))
