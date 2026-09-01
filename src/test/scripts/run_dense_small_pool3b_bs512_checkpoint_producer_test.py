from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_small_pool3b_checkpoint_evaluator as evaluator  # noqa: I001
import run_dense_small_pool3b_checkpoint_producer as producer


MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "models"
    / "manifests"
    / "dense-small-pool3b-bs512-checkpoint-producers-v1.json"
)


def argument(arguments: list[str], prefix: str) -> str:
    values = [value for value in arguments if value.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_bs512_manifest_has_four_one_node_gradient_accumulation_four_coordinates() -> None:
    config = producer.load_manifest(MANIFEST)
    assert len(config["producerCoordinates"]) == 4
    assert {(item["model"], item["weightDecay"]) for item in config["producerCoordinates"]} == {
        ("474m", "0.1"),
        ("474m", "0.3"),
        ("153m", "0.1"),
        ("153m", "0.3"),
    }
    for item in config["producerCoordinates"]:
        producer.validate_coordinate(config, item, check_filesystem=False)
        assert item["batchSequences"] == 512
        assert item["gpuCount"] == 8
        assert item["rankMicrobatchSequences"] == 16
        assert item["gradientAccumulationSteps"] == 4
        assert Path(item["sourceCheckpoint"]) == producer.expected_source(item)
        assert Path(item["output"]) == producer.expected_output(item)
        arguments = producer.base_arguments(item, str(config["repeatedManifest"]))
        assert argument(arguments, "--data_loader.global_batch_size=") == (
            f"--data_loader.global_batch_size={512 * producer.SEQUENCE_LENGTH}"
        )
        assert argument(arguments, "--train_module.rank_microbatch_size=") == (
            f"--train_module.rank_microbatch_size={16 * producer.SEQUENCE_LENGTH}"
        )


def test_bs512_checkpoint_and_evaluator_ladders_match_model_size() -> None:
    config = producer.load_manifest(MANIFEST)
    for item in config["producerCoordinates"]:
        increment = 16 if item["model"] == "474m" else 32
        epochs = producer.target_epochs(item, int(config["maxEpoch"]))
        assert epochs == list(range(increment, 257, increment))
        _, loaded = evaluator.load(MANIFEST, item["id"], increment)
        assert loaded == item
        assert evaluator.source_checkpoint(item, increment) == (
            Path(item["output"])
            / f"step{producer.stable_step(increment, producer.TARGET_POOL_TOKENS, 512)}"
        )
