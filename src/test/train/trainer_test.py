from types import SimpleNamespace
from unittest.mock import Mock

from olmo_core.train.common import LoadStrategy
from olmo_core.train.trainer import Trainer


def test_prefer_explicit_checkpoint_does_not_fall_back_to_latest_endpoint():
    trainer = object.__new__(Trainer)
    trainer.no_checkpoints = False
    trainer._checkpoint_loaded = False
    trainer.load_strategy = LoadStrategy.if_available
    trainer.prefer_explicit_load_path = True
    trainer.load_path = "/checkpoints/step858"
    trainer.save_folder = "/checkpoints"  # Also contains the newer step954 endpoint.
    trainer.reset_data_loader_state_on_load_path = False
    trainer.maybe_load_checkpoint = Mock()
    trainer.train_module = SimpleNamespace(validate_load_path_checkpoint=Mock())

    def load_checkpoint(path, **kwargs):
        assert path == "/checkpoints/step858"
        assert kwargs == {"reset_data_loader_state": False}
        trainer._checkpoint_loaded = True

    trainer.load_checkpoint = Mock(side_effect=load_checkpoint)

    trainer._load_initial_checkpoint()

    trainer.load_checkpoint.assert_called_once_with(
        "/checkpoints/step858", reset_data_loader_state=False
    )
    trainer.maybe_load_checkpoint.assert_not_called()
    trainer.train_module.validate_load_path_checkpoint.assert_called_once_with()
