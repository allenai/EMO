from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_474m_pool3b_bs256_e96_continuation as continuation  # noqa: I001


MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "models"
    / "manifests"
    / "dense-474m-pool3b-bs256-e96-continuation-v1.json"
)


def argument(arguments: list[str], prefix: str) -> str:
    values = [value for value in arguments if value.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_manifest_pins_exact_e96_lineage() -> None:
    config, item = continuation.load(MANIFEST)
    continuation.validate(config, item, check_filesystem=False)
    assert continuation.checkpoint_step(96) == 247192
    assert continuation.checkpoint_step(112) == 288391
    assert continuation.checkpoint_step(256) == 659179
    assert Path(item["sourceCheckpoint"]).parent == Path(item["output"])


def test_producer_loads_e96_and_retains_e112_through_e256() -> None:
    config, item = continuation.load(MANIFEST)
    source = Path(item["sourceCheckpoint"])
    arguments = continuation.producer_arguments(
        config, item, source, list(continuation.TARGET_EPOCHS)
    )
    assert argument(arguments, "--trainer.load_path=").endswith("/step247192")
    assert argument(arguments, "--trainer.max_duration=") == (
        "--trainer.max_duration={value: 659179, unit: steps}"
    )
    fixed = argument(arguments, "--trainer.callbacks.checkpointer.fixed_steps=")
    assert fixed.startswith("--trainer.callbacks.checkpointer.fixed_steps=[288391,")
    assert fixed.endswith(",659179]")
    assert "ConstantScheduler" in argument(arguments, "--train_module.scheduler=")
    assert not any("WSD" in value for value in arguments)
    assert "--trainer.reset_data_loader_state_on_load_path=true" in arguments


def test_future_evaluations_match_every_retained_epoch() -> None:
    assert continuation.EVALUATION_EPOCHS == continuation.TARGET_EPOCHS
