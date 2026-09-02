from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_small_pool3b_bs256_e384_continuation as continuation  # noqa: I001


MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "models"
    / "manifests"
    / "dense-small-pool3b-bs256-e384-continuation-v1.json"
)


def argument(arguments: list[str], prefix: str) -> str:
    values = [value for value in arguments if value.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_manifest_pins_exact_selected_recovery_lineage() -> None:
    config, item = continuation.load(MANIFEST)
    continuation.validate(config, item, check_filesystem=False)
    assert continuation.checkpoint_step(256) == 659179
    assert continuation.checkpoint_step(320) == 823974
    assert continuation.checkpoint_step(384) == 988769
    assert Path(item["sourceCheckpoint"]).parent == Path(item["output"])


def test_producer_is_constant_lr_and_retains_only_e320_e384() -> None:
    config, item = continuation.load(MANIFEST)
    arguments = continuation.producer_arguments(config, item)
    assert argument(arguments, "--trainer.load_path=").endswith("/step659179")
    assert argument(arguments, "--trainer.max_duration=") == (
        "--trainer.max_duration={value: 988769, unit: steps}"
    )
    assert argument(arguments, "--trainer.callbacks.checkpointer.fixed_steps=") == (
        "--trainer.callbacks.checkpointer.fixed_steps=[823974,988769]"
    )
    assert "ConstantScheduler" in argument(arguments, "--train_module.scheduler=")
    assert not any("WSD" in value for value in arguments)
    assert "--trainer.reset_data_loader_state_on_load_path=true" in arguments


def test_only_e320_and_e384_evaluations_are_authorized() -> None:
    assert continuation.EVALUATION_EPOCHS == (320, 384)
