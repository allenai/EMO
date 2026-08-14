from types import SimpleNamespace

from olmo_core.script_utils import _defer_checkpoint_selection_to_trainer


def test_explicit_checkpoint_preference_defers_save_folder_recovery_to_trainer():
    trainer = SimpleNamespace(
        prefer_explicit_load_path=True,
        load_path="/checkpoints/step858",
    )

    assert _defer_checkpoint_selection_to_trainer(trainer)


def test_default_checkpoint_policy_keeps_legacy_preload_behavior():
    trainer = SimpleNamespace(
        prefer_explicit_load_path=False,
        load_path="/checkpoints/step858",
    )

    assert not _defer_checkpoint_selection_to_trainer(trainer)
    trainer.prefer_explicit_load_path = True
    trainer.load_path = None
    assert not _defer_checkpoint_selection_to_trainer(trainer)
