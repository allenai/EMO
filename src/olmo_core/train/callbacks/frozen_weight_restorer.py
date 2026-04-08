"""
Callback to restore frozen expert weights after each optimizer step.

This is necessary because AdamW applies weight decay directly to weights,
bypassing the gradient masking mechanism used for partial freezing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

import torch

from .callback import Callback

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
    :param num_experts_to_train: Number of experts to train (from the end).
        The first (num_experts - num_experts_to_train) experts will be frozen.
    :param layer_patterns: List of parameter name patterns to match for freezing.
        Defaults to ["experts", "router"].
    :param restore_interval: How often to restore weights. Default 1 (every step).
        Set higher to reduce overhead if small drift is acceptable.
    :param log_drift: If True, logs the drift (change) in frozen weights before restoring.
    """

    priority: ClassVar[int] = -100  # Run very late, after optimizer step completes

    num_experts: int = 128
    num_experts_to_train: int = 1
    layer_patterns: List[str] = field(default_factory=lambda: ["experts", "router"])
    restore_interval: int = 1
    log_drift: bool = False

    # Internal state (not serialized)
    _frozen_weights: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def _get_frozen_slice(self, param: torch.Tensor) -> slice:
        """Get the slice for frozen portion of a parameter."""
        num_frozen = self.num_experts - self.num_experts_to_train
        expert_size = param.shape[0] // self.num_experts
        return slice(0, expert_size * num_frozen)

    def _should_freeze(self, name: str) -> bool:
        """Check if a parameter should have partial freezing applied."""
        return any(pattern in name for pattern in self.layer_patterns)

    def _save_frozen_weights(self):
        """Save the frozen portions of all matching parameters."""
        self._frozen_weights.clear()

        for name, param in self.trainer.train_module.model.named_parameters():
            if not self._should_freeze(name):
                continue

            frozen_slice = self._get_frozen_slice(param)
            # Clone and detach to avoid keeping computation graph
            # Use .cpu() to save GPU memory if needed (optional)
            self._frozen_weights[name] = param.data[frozen_slice].clone()

        log.info(
            f"FrozenWeightRestorer: Saved frozen weights for {len(self._frozen_weights)} parameters"
        )
        for name in list(self._frozen_weights.keys())[:3]:
            frozen_slice = self._get_frozen_slice(self._frozen_weights[name])
            log.info(f"  - {name}: shape {self._frozen_weights[name].shape}")
        if len(self._frozen_weights) > 3:
            log.info(f"  ... and {len(self._frozen_weights) - 3} more")

    def _restore_frozen_weights(self):
        """Restore the frozen portions of all matching parameters."""
        total_drift = 0.0
        num_restored = 0

        for name, param in self.trainer.train_module.model.named_parameters():
            if name not in self._frozen_weights:
                continue

            frozen_slice = self._get_frozen_slice(param)
            saved_weights = self._frozen_weights[name]

            if self.log_drift:
                # Calculate drift before restoring
                current_weights = param.data[frozen_slice]
                drift = (
                    (current_weights - saved_weights.to(current_weights.device)).abs().max().item()
                )
                total_drift = max(total_drift, drift)

            # Restore frozen weights
            with torch.no_grad():
                param.data[frozen_slice].copy_(saved_weights.to(param.device))

            num_restored += 1

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
        """Save frozen weights to checkpoint."""
        return {
            "frozen_weights": {k: v.cpu() for k, v in self._frozen_weights.items()},
            "initialized": self._initialized,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load frozen weights from checkpoint."""
        if "frozen_weights" in state_dict:
            self._frozen_weights = state_dict["frozen_weights"]
            self._initialized = state_dict.get("initialized", True)
            log.info(
                f"FrozenWeightRestorer: Loaded frozen weights for "
                f"{len(self._frozen_weights)} parameters from checkpoint"
            )
