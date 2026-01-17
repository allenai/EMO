"""
Callback for converting the final checkpoint to HuggingFace format at the end of training.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.utils import get_rank

from .callback import Callback
from .checkpointer import CheckpointerCallback

log = logging.getLogger(__name__)


@dataclass
class HFConverterCallback(Callback):
    """
    Converts the final saved checkpoint to HuggingFace format at the end of a training job.

    This callback runs after training completes and uses the conversion functions from
    the huggingface conversion script to convert the final OLMo Core checkpoint to
    a HuggingFace-compatible format.

    .. note::
        This callback requires the ``transformers`` library to be installed.
    """

    priority: ClassVar[int] = -1  # Run after checkpointer callback

    enabled: bool = True
    """
    Whether this callback is enabled. Set to ``False`` to disable HF conversion.
    """

    output_folder: Optional[str] = None
    """
    The folder to save the HuggingFace checkpoint to. If not specified, defaults to
    ``{checkpoint_path}/hf`` where ``checkpoint_path`` is the final checkpoint path.
    """

    dtype: Optional[DType] = DType.bfloat16
    """
    The dtype to save the HuggingFace model weights as. Defaults to bfloat16.
    """

    validate: bool = False
    """
    Whether to validate the converted model against the original model.
    Validation loads both models and compares their outputs.
    """

    debug: bool = False
    """
    Whether to output debug information during validation.
    Only has an effect if ``validate`` is True.
    """

    tokenizer_id: Optional[str] = None
    """
    The HuggingFace tokenizer identifier to save with the model.
    If not specified, uses the tokenizer from the experiment config.
    """

    max_sequence_length: Optional[int] = None
    """
    The maximum sequence length for the model. If not specified, uses the tokenizer's
    default max length.
    """

    device: Optional[str] = None
    """
    The device to use for conversion. Defaults to CPU.
    """

    moe_capacity_factor: Optional[float] = None
    """
    The MoE capacity factor. Higher values can decrease validation false negatives
    but may cause OOM errors. Only relevant for MoE models.
    """

    def _get_checkpointer_callback(self) -> Optional[CheckpointerCallback]:
        """Get the checkpointer callback from the trainer."""
        for callback in self.trainer.callbacks.values():
            if isinstance(callback, CheckpointerCallback):
                return callback
        return None

    def _get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        checkpointer = self._get_checkpointer_callback()
        if checkpointer is None:
            log.warning("CheckpointerCallback not found, cannot determine latest checkpoint path")
            return None

        if checkpointer._latest_checkpoint_path:
            return checkpointer._latest_checkpoint_path

        # Fallback: check if there are any saved checkpoints
        if checkpointer._checkpoints:
            return checkpointer._checkpoints[-1]

        return None

    def post_train(self):
        if not self.enabled:
            log.info("HFConverterCallback is disabled, skipping conversion")
            return

        # Only run on rank 0
        if get_rank() != 0:
            return

        checkpoint_path = self._get_latest_checkpoint_path()
        if checkpoint_path is None:
            log.warning("No checkpoint found, skipping HF conversion")
            return

        log.info(f"Converting checkpoint at '{checkpoint_path}' to HuggingFace format")

        # Determine output path
        if self.output_folder is not None:
            output_path = self.output_folder
        else:
            output_path = checkpoint_path + "-hf"  # join_path(checkpoint_path, "hf")

        # Import and call the conversion function
        try:
            from olmo_core.hf_utils import convert_checkpoint_to_hf, load_config
        except ImportError:
            log.error(
                "Failed to import conversion functions. Make sure that transformers library is installed."
            )
            return

        # Load config from checkpoint
        try:
            experiment_config = load_config(checkpoint_path)
        except Exception as e:
            log.error(f"Failed to load config from checkpoint: {e}")
            return

        if experiment_config is None:
            log.error("Experiment config not found in checkpoint, cannot convert to HF format")
            return

        transformer_config_dict = experiment_config.get("model")
        tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

        if transformer_config_dict is None:
            log.error("Model config not found in experiment config, cannot convert to HF format")
            return

        if tokenizer_config_dict is None:
            log.warning(
                "Tokenizer config not found in experiment config, "
                "conversion will proceed without tokenizer"
            )
            tokenizer_config_dict = {}

        # Determine device
        device = torch.device(self.device) if self.device else None

        try:
            convert_checkpoint_to_hf(
                original_checkpoint_path=checkpoint_path,
                output_path=output_path,
                transformer_config_dict=transformer_config_dict,
                tokenizer_config_dict=tokenizer_config_dict,
                dtype=self.dtype,
                tokenizer_id=self.tokenizer_id,
                max_sequence_length=self.max_sequence_length,
                validate=self.validate,
                debug=self.debug,
                device=device,
                moe_capacity_factor=self.moe_capacity_factor,
            )
            log.info(f"Successfully converted checkpoint to HuggingFace format at '{output_path}'")
        except Exception as e:
            log.error(f"Failed to convert checkpoint to HuggingFace format: {e}")
            raise
