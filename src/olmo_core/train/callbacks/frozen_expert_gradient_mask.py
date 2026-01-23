"""
Callback to mask gradients for frozen experts before optimizer step.

This is more robust than gradient hooks because it works with:
- torch.compile
- FSDP (including FSDP2 with DTensor)
- Any distributed training setup
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import torch

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class FrozenExpertGradientMaskCallback(Callback):
    """
    Masks gradients for frozen experts before each optimizer step.

    This callback runs at `pre_optim_step`, after the backward pass but before
    the optimizer updates weights. It zeros out gradients for frozen portions
    of expert parameters.

    This approach is more robust than `register_hook` because:
    1. Works with torch.compile (hooks can be broken by compilation)
    2. Works with FSDP (handles sharded parameters correctly)
    3. Runs at a well-defined point in the training loop

    :param num_experts: Total number of experts in the model.
    :param num_experts_to_train: Number of experts to train (from the end).
        The first (num_experts - num_experts_to_train) experts will be frozen.
    :param layer_patterns: List of parameter name patterns to match for freezing.
        Defaults to ["experts", "router"].
    """

    priority: ClassVar[int] = 100  # Run early in pre_optim_step, before gradient clipping

    num_experts: int = 128
    num_experts_to_train: int = 1
    layer_patterns: List[str] = field(default_factory=lambda: ["experts", "router"])

    # Internal state
    _mask_cache: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _logged_params: bool = field(default=False, repr=False)

    def _should_mask(self, name: str) -> bool:
        """Check if a parameter should have gradient masking applied."""
        return any(pattern in name for pattern in self.layer_patterns)

    def _get_or_create_mask(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Get or create a gradient mask for the given parameter."""
        cache_key = f"{name}_{param.shape}_{param.device}"

        if cache_key not in self._mask_cache:
            # Create mask: 1.0 for trainable, 0.0 for frozen
            mask = torch.ones_like(param, dtype=param.dtype)

            num_frozen = self.num_experts - self.num_experts_to_train
            expert_size = param.shape[0] // self.num_experts

            # Zero out frozen expert portion
            if num_frozen > 0:
                mask[:expert_size * num_frozen] = 0.0

            self._mask_cache[cache_key] = mask

        return self._mask_cache[cache_key]

    def pre_optim_step(self):
        """Zero out gradients for frozen experts before optimizer step."""
        masked_count = 0
        total_frozen_grad_norm = 0.0

        for name, param in self.trainer.train_module.model.named_parameters():
            if not self._should_mask(name):
                continue

            if param.grad is None:
                continue

            # Get or create the mask
            mask = self._get_or_create_mask(name, param.grad)

            # Calculate frozen gradient norm before masking (for debugging)
            num_frozen = self.num_experts - self.num_experts_to_train
            expert_size = param.grad.shape[0] // self.num_experts
            frozen_grad = param.grad[:expert_size * num_frozen]
            frozen_norm = frozen_grad.norm().item()
            total_frozen_grad_norm += frozen_norm

            # Apply mask in-place
            param.grad.mul_(mask)
            masked_count += 1

        # Log on first step and periodically
        if not self._logged_params:
            log.info(
                f"FrozenExpertGradientMask: Masking gradients for {masked_count} parameters "
                f"(freezing {self.num_experts - self.num_experts_to_train}/{self.num_experts} experts)"
            )
            self._logged_params = True

        # Log frozen gradient norm at logging intervals (should be ~0 if working correctly)
        if self.step % self.trainer.metrics_collect_interval == 0:
            if total_frozen_grad_norm > 1e-6:
                log.warning(
                    f"Step {self.step}: Frozen expert gradients had norm {total_frozen_grad_norm:.6e} "
                    f"before masking (now zeroed)"
                )
