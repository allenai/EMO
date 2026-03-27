"""
Callback for converting split-expert checkpoints back to regular format at the end of training.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.distributed.utils import barrier, get_rank

from .callback import Callback
from .checkpointer import CheckpointerCallback

log = logging.getLogger(__name__)


@dataclass
class SplitExpertConverterCallback(Callback):
    """
    Converts the final split-expert checkpoint back to regular format at the end of training.

    When training with :class:`~olmo_core.nn.moe.mlp.SplitExpertDroplessMoEMLP`, checkpoints
    contain split keys (``w1_frozen``, ``w1_trainable``, etc.). This callback gathers the full
    model state dict, merges split expert weights back into ``w1``, ``w2``, ``w3``, and saves
    a regular-format checkpoint.

    .. warning::
        In distributed training, ALL ranks must participate because gathering the full model
        state dict from FSDP requires collective operations. Only rank 0 saves.
    """

    priority: ClassVar[int] = 0  # Run before HF converter (-1) so it can use the regular checkpoint

    enabled: bool = True

    experts_to_train: List[int] = field(default_factory=list)
    """Expert indices that were trainable (needed to reconstruct full tensors)."""

    output_folder: Optional[str] = None
    """
    Folder to save the regular checkpoint. Defaults to ``{checkpoint_path}_regular``.
    """

    def _get_latest_checkpoint_path(self) -> Optional[str]:
        for callback in self.trainer.callbacks.values():
            if isinstance(callback, CheckpointerCallback):
                if callback._latest_checkpoint_path:
                    return callback._latest_checkpoint_path
                if callback._checkpoints:
                    return callback._checkpoints[-1]
        return None

    def _get_full_model_state_dict(self) -> Dict[str, Any]:
        """Collective operation — ALL ranks must call this."""
        model = self.trainer.train_module.model
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        return dist_cp_sd.get_model_state_dict(model, options=sd_options)

    def _merge_split_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Merge w1_frozen/w1_trainable → w1 (and w2, w3) in the state dict."""
        # Get expert layout from the model
        moe_mlp = self.trainer.train_module.model.blocks[0].feed_forward_moe.experts.mlp
        num_experts = moe_mlp.num_experts
        hidden_size = moe_mlp.hidden_size
        frozen_indices = sorted(set(range(num_experts)) - set(self.experts_to_train))

        frozen_rows = []
        for idx in frozen_indices:
            frozen_rows.extend(range(idx * hidden_size, (idx + 1) * hidden_size))
        trainable_rows = []
        for idx in sorted(self.experts_to_train):
            trainable_rows.extend(range(idx * hidden_size, (idx + 1) * hidden_size))
        frozen_row_idx = torch.tensor(frozen_rows, dtype=torch.long)
        trainable_row_idx = torch.tensor(trainable_rows, dtype=torch.long)

        result = {}
        handled = set()
        for key in state_dict:
            for wname in ("w1", "w2", "w3"):
                frozen_suffix = f"{wname}_frozen"
                trainable_suffix = f"{wname}_trainable"
                if key.endswith(frozen_suffix):
                    prefix = key[: -len(frozen_suffix)]
                    trainable_key = f"{prefix}{trainable_suffix}"
                    if trainable_key in state_dict:
                        d_model = state_dict[key].shape[1]
                        full = torch.empty(
                            num_experts * hidden_size, d_model, dtype=state_dict[key].dtype
                        )
                        full[frozen_row_idx] = state_dict[key]
                        full[trainable_row_idx] = state_dict[trainable_key]
                        result[f"{prefix}{wname}"] = full
                        handled.add(key)
                        handled.add(trainable_key)

        for key, val in state_dict.items():
            if key not in handled:
                result[key] = val
        return result

    def post_train(self):
        if not self.enabled:
            barrier()
            return

        if not self.experts_to_train:
            log.warning("No experts_to_train specified, skipping split→regular conversion")
            barrier()
            return

        checkpoint_path = self._get_latest_checkpoint_path()
        if checkpoint_path is None:
            log.warning("No checkpoint found, skipping split→regular conversion")
            barrier()
            return

        # ALL ranks must participate in gathering the full state dict
        log.info("Gathering full model state dict for split→regular conversion...")
        try:
            model_state_dict = self._get_full_model_state_dict()
        except Exception as e:
            log.error(f"Failed to get model state dict: {e}")
            barrier()
            raise

        # Merge on rank 0 (state dict is only populated there with full_state_dict=True)
        merged_sd = None
        if get_rank() == 0:
            merged_sd = self._merge_split_state_dict(model_state_dict)

            output_path = self.output_folder or f"{checkpoint_path}_regular"
            log.info(f"Merging split expert weights and saving to '{output_path}'")

            os.makedirs(output_path, exist_ok=True)
            torch.save(merged_sd, os.path.join(output_path, "model.pt"))

            # Also copy config.json if it exists in the checkpoint
            config_src = os.path.join(checkpoint_path, "config.json")
            if os.path.exists(config_src):
                import shutil
                shutil.copy2(config_src, os.path.join(output_path, "config.json"))

            log.info(f"Regular checkpoint saved to {output_path}")

        # Store merged state dict so HFConverterCallback can reuse it
        self.merged_model_state_dict = merged_sd

        barrier()
