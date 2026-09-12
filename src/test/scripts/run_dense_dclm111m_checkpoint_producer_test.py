from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_dclm111m_checkpoint_producer as producer  # noqa: I001


MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "models"
    / "manifests"
    / "dense-dclm111m-checkpoint-producers-v1.json"
)


def test_474m_bs64_wd1_continues_to_saturation_with_original_cadence() -> None:
    producer.configure_base()
    config = producer.load_manifest(MANIFEST)
    item = producer.coordinate_for_target(
        config, producer.BS64_474M_WD1_CONTINUATION_ID, 96
    )

    assert item["retainedCheckpointEpochs"] == list(range(4, 97, 4))
    assert item["evaluationEpochs"] == list(range(8, 97, 8))
    assert item["continuationSourceEpoch"] == 32
    assert item["continuationTargetEpoch"] == 96
    assert item["maxEpoch"] == 96
    assert item["gpuCount"] == 4
    assert item["rankMicrobatchSequences"] == 16
    producer.validate_coordinate(item)


def test_474m_pool111m_continuation_is_restricted_to_requested_coordinate() -> None:
    producer.configure_base()
    config = producer.load_manifest(MANIFEST)

    with pytest.raises(ValueError, match="does not authorize continuation target E96"):
        producer.coordinate_for_target(
            config, "dense-474m-dclm111m-bs64-lr2e-3-wd0.3", 96
        )


def test_474m_bs64_lr1e3_wd1_is_an_independent_valid_coordinate() -> None:
    producer.configure_base()
    config = producer.load_manifest(MANIFEST)
    item = producer.coordinate_for_target(
        config, "dense-474m-dclm111m-bs64-lr1e-3-wd1.0", None
    )

    assert item["retainedCheckpointEpochs"] == list(range(4, 33, 4))
    assert item["evaluationEpochs"] == [8, 16, 24, 32]
    assert item["gpuCount"] == 4
    assert item["rankMicrobatchSequences"] == 16
    producer.validate_coordinate(item)


def test_153m_bs64_lr2e3_wd03_explicitly_continues_from_e32_to_e48() -> None:
    producer.configure_base()
    config = producer.load_manifest(MANIFEST)
    item = producer.coordinate_for_target(
        config, "dense-153m-dclm111m-bs64-lr2e-3-wd0.3", 48
    )

    assert item["retainedCheckpointEpochs"] == [8, 16, 24, 32, 40, 48]
    assert item["evaluationEpochs"] == [16, 32, 48]
    assert item["continuationSourceEpoch"] == 32
    assert item["continuationTargetEpoch"] == 48
    assert item["stopOnAdjacentPostNonImprovement"] is False
    producer.validate_coordinate(item)
