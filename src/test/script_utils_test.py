from types import SimpleNamespace

from olmo_core.script_utils import _defer_checkpoint_selection_to_trainer


def test_explicit_checkpoint_preference_defers_save_folder_recovery_to_trainer():
    trainer = SimpleNamespace(prefer_explicit_load_path=True)

    assert _defer_checkpoint_selection_to_trainer(trainer, "/checkpoints/step858")


def test_default_checkpoint_policy_keeps_legacy_preload_behavior():
    trainer = SimpleNamespace(prefer_explicit_load_path=False)

    assert not _defer_checkpoint_selection_to_trainer(trainer, "/checkpoints/step858")
    trainer.prefer_explicit_load_path = True
    assert not _defer_checkpoint_selection_to_trainer(trainer, None)
