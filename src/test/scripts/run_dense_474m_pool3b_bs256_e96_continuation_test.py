from __future__ import annotations

import json
import sys
import tempfile
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


def make_complete_checkpoint(path: Path) -> None:
    (path / "model_and_optim").mkdir(parents=True)
    (path / "model_and_optim" / ".metadata").touch()
    (path / "model_and_optim" / "rank0.distcp").touch()
    (path / "train").mkdir()
    (path / "config.json").write_text("{}")
    for rank in range(8):
        (path / "train" / f"rank{rank}.pt").touch()


def test_manifest_pins_exact_e96_lineage() -> None:
    config, item = continuation.load(MANIFEST)
    continuation.validate(config, item, check_filesystem=False)
    assert continuation.checkpoint_step(96) == 247192
    assert continuation.checkpoint_step(112) == 288390
    assert continuation.checkpoint_step(256) == 659179
    assert Path(item["sourceCheckpoint"]).parent == Path(item["output"])


def test_producer_loads_e96_and_checkpoints_every_two_epochs() -> None:
    config, item = continuation.load(MANIFEST)
    source = Path(item["sourceCheckpoint"])
    arguments = continuation.producer_arguments(
        config, item, source, list(continuation.CHECKPOINT_EPOCHS)
    )
    assert argument(arguments, "--trainer.load_path=").endswith("/step247192")
    assert argument(arguments, "--trainer.max_duration=") == (
        "--trainer.max_duration={value: 659179, unit: steps}"
    )
    fixed = json.loads(
        argument(arguments, "--trainer.callbacks.checkpointer.fixed_steps=").split("=", 1)[1]
    )
    assert fixed == [
        continuation.checkpoint_step(epoch) for epoch in continuation.CHECKPOINT_EPOCHS
    ]
    assert continuation.CHECKPOINT_EPOCHS == tuple(range(98, 257, 2))
    assert "ConstantScheduler" in argument(arguments, "--train_module.scheduler=")
    assert not any("WSD" in value for value in arguments)
    assert "--trainer.reset_data_loader_state_on_load_path=true" in arguments


def test_future_evaluations_keep_the_original_sixteen_epoch_cadence() -> None:
    assert continuation.EVALUATION_EPOCHS == continuation.TARGET_EPOCHS
    assert continuation.CLEANUP_EPOCHS == tuple(
        epoch
        for epoch in continuation.CHECKPOINT_EPOCHS
        if epoch not in continuation.EVALUATION_EPOCHS
    )


def test_cleanup_deletes_only_recovery_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as raw:
        output = Path(raw)
        recovery = output / f"step{continuation.checkpoint_step(98)}"
        essential = output / f"step{continuation.checkpoint_step(112)}"
        make_complete_checkpoint(recovery)
        make_complete_checkpoint(essential)
        original_output = continuation.EXPECTED_OUTPUT
        original_cleanup = continuation.CLEANUP_EPOCHS
        original_evaluations = continuation.EVALUATION_EPOCHS
        try:
            continuation.EXPECTED_OUTPUT = output
            continuation.CLEANUP_EPOCHS = (98,)
            continuation.EVALUATION_EPOCHS = (112,)
            removed = continuation.cleanup_nonessential_checkpoints(output / ".state")
        finally:
            continuation.EXPECTED_OUTPUT = original_output
            continuation.CLEANUP_EPOCHS = original_cleanup
            continuation.EVALUATION_EPOCHS = original_evaluations
        assert removed == [98]
        assert not recovery.exists()
        assert essential.is_dir()
