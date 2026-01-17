"""Tests for HFConverterCallback."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.callbacks.hf_converter import HFConverterCallback


def _hf_conversion_available() -> bool:
    """Check if HF conversion dependencies are available."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        from examples.huggingface.convert_checkpoint_to_hf import (
            convert_checkpoint_to_hf,
        )

        return True
    except ImportError:
        return False


requires_hf_conversion = pytest.mark.skipif(
    not _hf_conversion_available(),
    reason="HF conversion dependencies not available",
)


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture
def transformer_config(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    return TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size())


@pytest.fixture
def checkpoint_with_config(
    tmp_path: Path,
    transformer_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
) -> Path:
    """Create a checkpoint directory with config.json and model weights."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    # Save model weights
    model = transformer_config.build()
    save_model_and_optim_state(checkpoint_path / "model_and_optim", model)

    # Save config.json (simulating what ConfigSaverCallback does)
    config = {
        "model": transformer_config.as_config_dict(),
        "dataset": {
            "tokenizer": tokenizer_config.as_config_dict(),
        },
    }
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(config, f)

    return checkpoint_path


@pytest.fixture
def mock_trainer(checkpoint_with_config: Path) -> MagicMock:
    """Create a mock trainer with a checkpointer callback."""
    trainer = MagicMock()

    # Create a real checkpointer callback and set its internal state
    checkpointer_callback = CheckpointerCallback()
    checkpointer_callback._latest_checkpoint_path = str(checkpoint_with_config)
    checkpointer_callback._checkpoints = [str(checkpoint_with_config)]
    checkpointer_callback._trainer = trainer

    trainer.callbacks = {"checkpointer": checkpointer_callback}

    return trainer


@pytest.fixture
def simple_mock_trainer() -> MagicMock:
    """Create a simple mock trainer without checkpoint fixtures (for basic tests)."""
    trainer = MagicMock()

    checkpointer_callback = CheckpointerCallback()
    checkpointer_callback._latest_checkpoint_path = "/fake/path/checkpoint"
    checkpointer_callback._checkpoints = ["/fake/path/checkpoint"]
    checkpointer_callback._trainer = trainer

    trainer.callbacks = {"checkpointer": checkpointer_callback}

    return trainer


class TestHFConverterCallback:
    def test_disabled_callback_skips_conversion(self, simple_mock_trainer: MagicMock):
        """Test that a disabled callback doesn't run conversion."""
        callback = HFConverterCallback(enabled=False)
        callback._trainer = simple_mock_trainer

        # When disabled, post_train should return early without trying to import/call conversion
        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=0):
            # This should not raise any errors - it returns early due to enabled=False
            callback.post_train()

    def test_non_rank_zero_skips_conversion(self, simple_mock_trainer: MagicMock):
        """Test that non-rank-0 processes don't run conversion."""
        callback = HFConverterCallback(enabled=True)
        callback._trainer = simple_mock_trainer

        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=1):
            # This should not raise any errors - it returns early due to rank != 0
            callback.post_train()

    def test_get_checkpointer_callback(self, simple_mock_trainer: MagicMock):
        """Test that the callback can find the checkpointer callback."""
        callback = HFConverterCallback()
        callback._trainer = simple_mock_trainer

        checkpointer = callback._get_checkpointer_callback()
        assert checkpointer is not None
        assert isinstance(checkpointer, CheckpointerCallback)

    def test_get_checkpointer_callback_missing(self):
        """Test behavior when no checkpointer callback is found."""
        callback = HFConverterCallback()
        callback._trainer = MagicMock()
        callback._trainer.callbacks = {}

        checkpointer = callback._get_checkpointer_callback()
        assert checkpointer is None

    def test_get_latest_checkpoint_path(self, simple_mock_trainer: MagicMock):
        """Test that the callback can get the latest checkpoint path."""
        callback = HFConverterCallback()
        callback._trainer = simple_mock_trainer

        path = callback._get_latest_checkpoint_path()
        assert path == "/fake/path/checkpoint"

    def test_get_latest_checkpoint_path_from_checkpoints_list(self, simple_mock_trainer: MagicMock):
        """Test fallback to _checkpoints list when _latest_checkpoint_path is empty."""
        callback = HFConverterCallback()
        callback._trainer = simple_mock_trainer

        # Clear _latest_checkpoint_path but keep _checkpoints
        checkpointer = simple_mock_trainer.callbacks["checkpointer"]
        checkpointer._latest_checkpoint_path = ""

        path = callback._get_latest_checkpoint_path()
        assert path == "/fake/path/checkpoint"

    def test_get_latest_checkpoint_path_none(self, simple_mock_trainer: MagicMock):
        """Test behavior when no checkpoint is available."""
        callback = HFConverterCallback()
        callback._trainer = simple_mock_trainer

        # Clear both _latest_checkpoint_path and _checkpoints
        checkpointer = simple_mock_trainer.callbacks["checkpointer"]
        checkpointer._latest_checkpoint_path = ""
        checkpointer._checkpoints = []

        path = callback._get_latest_checkpoint_path()
        assert path is None

    def test_post_train_no_checkpointer_callback(self):
        """Test behavior when CheckpointerCallback is not present."""
        callback = HFConverterCallback(enabled=True)
        callback._trainer = MagicMock()
        callback._trainer.callbacks = {}

        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=0):
            # Should warn and return early
            callback.post_train()

    def test_callback_priority(self):
        """Test that the callback has lower priority than checkpointer."""
        hf_callback = HFConverterCallback()
        checkpointer_callback = CheckpointerCallback()

        # HF converter should run after checkpointer (lower priority)
        assert hf_callback.priority < checkpointer_callback.priority

    @requires_hf_conversion
    def test_post_train_converts_checkpoint(
        self,
        tmp_path: Path,
        checkpoint_with_config: Path,
        mock_trainer: MagicMock,
        transformer_config: TransformerConfig,
    ):
        """Test that post_train successfully converts the checkpoint to HF format."""
        from transformers import AutoConfig

        output_folder = tmp_path / "hf_output"

        callback = HFConverterCallback(
            enabled=True,
            output_folder=str(output_folder),
            validate=False,  # Skip validation for faster test
        )
        callback._trainer = mock_trainer

        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=0):
            callback.post_train()

        # Verify HF model was created
        assert output_folder.exists()
        hf_config = AutoConfig.from_pretrained(output_folder)
        assert hf_config.hidden_size == transformer_config.d_model
        assert hf_config.num_hidden_layers == transformer_config.n_layers

    @requires_hf_conversion
    def test_post_train_default_output_path(
        self,
        checkpoint_with_config: Path,
        mock_trainer: MagicMock,
    ):
        """Test that default output path is checkpoint_path-hf."""
        from transformers import AutoConfig

        callback = HFConverterCallback(
            enabled=True,
            output_folder=None,  # Use default
            validate=False,
        )
        callback._trainer = mock_trainer

        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=0):
            callback.post_train()

        # Verify HF model was created in default location
        expected_output = Path(str(checkpoint_with_config) + "-hf")
        assert expected_output.exists()
        hf_config = AutoConfig.from_pretrained(expected_output)
        assert hf_config is not None

    @requires_hf_conversion
    def test_post_train_model_correctness(
        self,
        tmp_path: Path,
        checkpoint_with_config: Path,
        mock_trainer: MagicMock,
        transformer_config: TransformerConfig,
    ):
        """Test that the converted HF model produces the same output as the original."""
        from transformers import AutoModelForCausalLM

        output_folder = tmp_path / "hf_output"

        callback = HFConverterCallback(
            enabled=True,
            output_folder=str(output_folder),
            validate=False,
            dtype=None,  # Preserve original dtype for accurate comparison
        )
        callback._trainer = mock_trainer

        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=0):
            callback.post_train()

        # Load original OLMo Core model
        olmo_core_model = transformer_config.build()
        load_model_and_optim_state(
            checkpoint_with_config / "model_and_optim", model=olmo_core_model
        )
        olmo_core_model.eval()

        # Load converted HF model
        hf_model = AutoModelForCausalLM.from_pretrained(output_folder)
        hf_model.eval()

        # Compare outputs
        min_vocab_size = min(int(hf_model.vocab_size), olmo_core_model.vocab_size)
        rand_input = torch.randint(0, min_vocab_size, (2, 8))

        with torch.no_grad():
            hf_logits = hf_model(input_ids=rand_input).logits
            olmo_core_logits = olmo_core_model(input_ids=rand_input)

        torch.testing.assert_close(
            hf_logits[..., :min_vocab_size],
            olmo_core_logits[..., :min_vocab_size],
            rtol=1e-4,
            atol=1e-4,
        )

    @requires_hf_conversion
    def test_post_train_missing_config(self, tmp_path: Path, mock_trainer: MagicMock):
        """Test behavior when config.json is missing from checkpoint."""
        # Create checkpoint without config.json
        checkpoint_path = tmp_path / "checkpoint_no_config"
        checkpoint_path.mkdir()
        (checkpoint_path / "model_and_optim").mkdir()

        # Update mock trainer to point to this checkpoint
        checkpointer = mock_trainer.callbacks["checkpointer"]
        checkpointer._latest_checkpoint_path = str(checkpoint_path)

        callback = HFConverterCallback(enabled=True, validate=False)
        callback._trainer = mock_trainer

        with patch("olmo_core.train.callbacks.hf_converter.get_rank", return_value=0):
            # Should log error but not raise (config loading will fail)
            callback.post_train()
