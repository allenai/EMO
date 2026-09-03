from __future__ import annotations

import json
import sys
import tempfile
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


def make_complete_checkpoint(path: Path) -> None:
    (path / "model_and_optim").mkdir(parents=True)
    (path / "model_and_optim" / ".metadata").touch()
    (path / "model_and_optim" / "rank0.distcp").touch()
    (path / "train").mkdir()
    (path / "config.json").write_text("{}")
    for rank in range(8):
        (path / "train" / f"rank{rank}.pt").touch()


def test_manifest_pins_exact_selected_recovery_lineage() -> None:
    config, item = continuation.load(MANIFEST)
    continuation.validate(config, item, check_filesystem=False)
    assert continuation.checkpoint_step(256) == 659179
    assert continuation.checkpoint_step(320) == 823974
    assert continuation.checkpoint_step(384) == 988769
    assert Path(item["sourceCheckpoint"]).parent == Path(item["output"])


def test_producer_is_constant_lr_and_checkpoints_every_four_epochs() -> None:
    config, item = continuation.load(MANIFEST)
    arguments = continuation.producer_arguments(
        config,
        item,
        Path(item["sourceCheckpoint"]),
        list(continuation.CHECKPOINT_EPOCHS),
    )
    assert argument(arguments, "--trainer.load_path=").endswith("/step659179")
    assert argument(arguments, "--trainer.max_duration=") == (
        "--trainer.max_duration={value: 988769, unit: steps}"
    )
    fixed = json.loads(
        argument(arguments, "--trainer.callbacks.checkpointer.fixed_steps=").split("=", 1)[1]
    )
    assert fixed == [
        continuation.checkpoint_step(epoch) for epoch in continuation.CHECKPOINT_EPOCHS
    ]
    assert continuation.CHECKPOINT_EPOCHS == tuple(range(260, 385, 4))
    assert "ConstantScheduler" in argument(arguments, "--train_module.scheduler=")
    assert not any("WSD" in value for value in arguments)
    assert "--trainer.reset_data_loader_state_on_load_path=true" in arguments


def test_only_e320_and_e384_evaluations_are_authorized() -> None:
    assert continuation.EVALUATION_EPOCHS == (320, 384)
    assert continuation.CLEANUP_EPOCHS == tuple(
        epoch
        for epoch in continuation.CHECKPOINT_EPOCHS
        if epoch not in continuation.EVALUATION_EPOCHS
    )


def test_cleanup_deletes_only_recovery_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as raw:
        output = Path(raw)
        recovery = output / f"step{continuation.checkpoint_step(260)}"
        essential = output / f"step{continuation.checkpoint_step(320)}"
        make_complete_checkpoint(recovery)
        make_complete_checkpoint(essential)
        original_cleanup = continuation.CLEANUP_EPOCHS
        original_evaluations = continuation.EVALUATION_EPOCHS
        try:
            continuation.CLEANUP_EPOCHS = (260,)
            continuation.EVALUATION_EPOCHS = (320,)
            removed = continuation.cleanup_nonessential_checkpoints(output, output / ".state")
        finally:
            continuation.CLEANUP_EPOCHS = original_cleanup
            continuation.EVALUATION_EPOCHS = original_evaluations
        assert removed == [260]
        assert not recovery.exists()
        assert essential.is_dir()
