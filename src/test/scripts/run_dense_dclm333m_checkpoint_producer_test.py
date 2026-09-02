from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_dclm333m_checkpoint_producer as producer  # noqa: I001
import submit_dense_dclm333m_checkpoint_producers as submitter  # noqa: I001


MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "models"
    / "manifests"
    / "dense-dclm333m-checkpoint-producers-v1.json"
)


def argument(arguments: list[str], prefix: str) -> str:
    values = [value for value in arguments if value.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_dense_1b_pool333m_first_stage_is_exactly_three_coordinates() -> None:
    config = producer.load_manifest(MANIFEST)
    items = [item for item in config["producerCoordinates"] if item["model"] == "1b"]
    assert {
        (item["batchSequences"], item["learningRate"], item["weightDecay"])
        for item in items
    } == {
        (32, "5e-4", "0.3"),
        (64, "1e-3", "0.3"),
        (64, "1e-3", "1.0"),
    }
    assert all(item["maxEpoch"] == 32 for item in items)
    assert all(item["retainedCheckpointEpochs"] == list(range(4, 33, 4)) for item in items)
    assert all(item["evaluationEpochs"] == [8, 16, 24, 32] for item in items)
    assert not any(item["batchSequences"] == 32 and item["weightDecay"] == "1.0" for item in items)


def test_dense_1b_pool333m_topology_and_isolated_post_sources() -> None:
    config = producer.load_manifest(MANIFEST)
    for item in config["producerCoordinates"][:3]:
        producer.validate_coordinate(item)
        assert producer.gpu_count(item) == 8
        assert item["rankMicrobatchSequences"] * producer.gpu_count(item) == item["batchSequences"]
        base = producer.base_arguments(item)
        assert argument(base, "--data_loader.global_batch_size=") == (
            f"--data_loader.global_batch_size={item['batchSequences'] * producer.SEQUENCE_LENGTH}"
        )
        assert argument(base, "--train_module.rank_microbatch_size=") == (
            f"--train_module.rank_microbatch_size={item['rankMicrobatchSequences'] * producer.SEQUENCE_LENGTH}"
        )
        source = Path(item["output"]) / f"step{producer.stable_step(8, item['batchSequences'])}"
        post_output = producer.state_dir(item) / "post_decay_runs" / "e8"
        post = producer.postdecay_arguments(item, 8, source, post_output, "test")
        assert f"--trainer.load_path={source}" in post
        assert f"--save-folder={post_output}" in post
        assert source.parent != post_output


def test_dense_1b_pool333m_min_runtime_respects_beaker_limit() -> None:
    config = producer.load_manifest(MANIFEST)
    for item in config["producerCoordinates"][:3]:
        assert submitter.reserved_min_runtime(item, config["runtimeEstimate"]) == 8 * 60 * 60
        assert (
            submitter.reserved_min_runtime(
                item, config["runtimeEstimate"], omitted=True
            )
            == 0
        )
