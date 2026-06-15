"""
Callback to restore frozen expert weights after each optimizer step.

This is necessary because AdamW applies weight decay directly to weights,
bypassing the gradient masking mechanism used for partial freezing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

from olmo_core.distributed.utils import get_local_tensor

from .callback import Callback
from .frozen_expert_gradient_mask import _create_expert_mask_1d

log = logging.getLogger(__name__)


@dataclass
class FrozenWeightRestorerCallback(Callback):
    """
    Restores frozen portions of expert weights after each optimizer step.

    When using partial freezing with gradient masks, the gradient-based updates
    are correctly masked. However, AdamW's weight decay is applied directly to
    weights (not through gradients), causing frozen weights to drift.

    This callback saves the frozen portions of weights before training and
    restores them after each optimizer step.

    :param num_experts: Total number of experts in the model.
    :param num_experts_to_train: Number of experts to train (the last k non-shared
        experts). All other experts (including the shared experts at the very end)
        are frozen and restored each step.
    :param num_shared_experts: Number of shared experts, which occupy the last
        ``num_shared_experts`` indices and are always frozen. Default 0 reduces to
        the original "freeze all but the last k experts" behavior.
    :param layer_patterns: List of parameter name patterns to match for freezing.
        Defaults to ["experts", "router"].
    :param restore_interval: How often to restore weights. Default 1 (every step).
        Set higher to reduce overhead if small drift is acceptable.
    :param log_drift: If True, logs the drift (change) in frozen weights before restoring.
    """

    priority: ClassVar[int] = -100  # Run very late, after optimizer step completes

    num_experts: int = 128
    num_experts_to_train: int = 1
    num_shared_experts: int = 0
    layer_patterns: List[str] = field(default_factory=lambda: ["experts", "router"])
    restore_interval: int = 1
    log_drift: bool = False

    # Internal state. Per matching param we keep a *local* (FSDP shard) trainable
    # mask (1.0 = trainable row, 0.0 = frozen) and a *local* snapshot of the frozen
    # rows (= local_weights * (1 - mask) at save time). Restore is the FSDP/compile-
    # safe elementwise op  local <- local * mask + frozen_snapshot: trainable rows
    # are kept, frozen rows are forced back to the snapshot.
    _train_mask: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _frozen_snapshot: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def _should_freeze(self, name: str) -> bool:
        """Check if a parameter should have partial freezing applied."""
        return any(pattern in name for pattern in self.layer_patterns)

    def _build_local_mask(self, param: torch.Tensor) -> torch.Tensor:
        """Local (FSDP-shard-aligned) float mask, 1.0 on trainable rows, broadcastable to param."""
        full_shape = param.shape
        full_mask = _create_expert_mask_1d(
            size=full_shape[0],
            num_experts=self.num_experts,
            num_experts_to_train=self.num_experts_to_train,
            dtype=param.dtype,
            device="cpu",
            num_shared_experts=self.num_shared_experts,
        )
        if len(full_shape) > 1:
            full_mask = full_mask.view(-1, *([1] * (len(full_shape) - 1)))
            full_mask = full_mask.expand(full_shape).contiguous()
        if isinstance(param.data, DTensor):
            local_dev = get_local_tensor(param.data).device
            mask_dt = distribute_tensor(
                full_mask.to(local_dev), param.data.device_mesh, param.data.placements
            )
            return get_local_tensor(mask_dt)
        return full_mask.to(param.device)

    def _save_frozen_weights(self):
        """Snapshot the frozen rows (local) of all matching parameters."""
        self._train_mask.clear()
        self._frozen_snapshot.clear()

        for name, param in self.trainer.train_module.model.named_parameters():
            if not self._should_freeze(name):
                continue
            mask = self._build_local_mask(param)
            local = get_local_tensor(param.data)
            self._train_mask[name] = mask
            self._frozen_snapshot[name] = (local * (1.0 - mask)).clone()

        log.info(
            f"FrozenWeightRestorer: Saved frozen weights for {len(self._frozen_snapshot)} parameters"
        )

    def _restore_frozen_weights(self):
        """Restore the frozen rows of all matching parameters (local, FSDP-safe)."""
        total_drift = 0.0

        for name, param in self.trainer.train_module.model.named_parameters():
            if name not in self._frozen_snapshot:
                continue

            mask = self._train_mask[name]
            frozen = self._frozen_snapshot[name]
            local = get_local_tensor(param.data)

            if self.log_drift:
                drift = ((local * (1.0 - mask)) - frozen).abs().max().item()
                total_drift = max(total_drift, drift)

            # local <- local * mask + frozen_snapshot
            with torch.no_grad():
                local.mul_(mask).add_(frozen)

        if self.log_drift and self.step % self.trainer.metrics_collect_interval == 0:
            log.info(f"Step {self.step}: Max frozen weight drift before restore: {total_drift:.6e}")

    def pre_train(self):
        """Save frozen weights before training starts."""
        self._save_frozen_weights()
        self._initialized = True

    def post_checkpoint_loaded(self, path):
        """Re-save frozen weights after loading a checkpoint."""
        # After loading checkpoint, we need to re-capture the frozen weights
        # in case they differ from what we saved initially
        if self._initialized:
            log.info("FrozenWeightRestorer: Re-saving frozen weights after checkpoint load")
            self._save_frozen_weights()

    def post_train_batch(self):
        """Restore frozen weights after each optimizer step."""
        if not self._initialized:
            return

        if self.step % self.restore_interval != 0:
            return

        self._restore_frozen_weights()

    def state_dict(self) -> Dict[str, Any]:
        # Only persist the init flag; the local frozen snapshot is rebuilt from the
        # (unchanged) frozen weights in post_checkpoint_loaded, which avoids tying the
        # serialized state to a particular FSDP world size / shard layout.
        return {"initialized": self._initialized}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._initialized = state_dict.get("initialized", False)
