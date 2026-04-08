"""
Callback to mask gradients for frozen experts before optimizer step.

This is more robust than gradient hooks because it works with:
- torch.compile
- FSDP (including FSDP2 with DTensor)
- Any distributed training setup
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, List

import torch
from torch.distributed.tensor import DTensor

from olmo_core.distributed.utils import get_local_tensor

from .callback import Callback

log = logging.getLogger(__name__)


def _create_expert_mask_1d(
    size: int,
    num_experts: int,
    num_experts_to_train: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a 1D mask for expert parameters.

    Returns a mask where frozen expert indices are 0.0 and trainable are 1.0.
    Assumes experts are stacked along dimension 0.
    """
    mask = torch.ones(size, dtype=dtype, device=device)
    num_frozen = num_experts - num_experts_to_train
    if num_frozen > 0:
        expert_size = size // num_experts
        frozen_end = expert_size * num_frozen
        mask[:frozen_end] = 0.0
    return mask


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
    _mask_cache: dict = field(default_factory=dict, repr=False)
    _logged_params: bool = field(default=False, repr=False)

    def _should_mask(self, name: str) -> bool:
        """Check if a parameter should have gradient masking applied."""
        return any(pattern in name for pattern in self.layer_patterns)

    def _get_or_create_mask(
        self, name: str, grad: torch.Tensor, full_shape: torch.Size
    ) -> torch.Tensor:
        """
        Get or create a gradient mask for the given parameter.

        Handles both regular tensors and DTensors (FSDP sharded).
        """
        # For DTensor, we need to work with the local tensor
        local_grad = get_local_tensor(grad)

        cache_key = f"{name}_{local_grad.shape}_{local_grad.device}"

        if cache_key not in self._mask_cache:
            if isinstance(grad, DTensor):
                # For sharded tensors, create full mask then redistribute
                # to match the gradient's sharding
                full_mask = _create_expert_mask_1d(
                    size=full_shape[0],
                    num_experts=self.num_experts,
                    num_experts_to_train=self.num_experts_to_train,
                    dtype=local_grad.dtype,
                    device="cpu",  # Create on CPU first
                )
                # Reshape to match full parameter shape
                if len(full_shape) > 1:
                    full_mask = full_mask.view(-1, *([1] * (len(full_shape) - 1)))
                    full_mask = full_mask.expand(full_shape).contiguous()

                # Distribute with same placement as gradient
                from torch.distributed.tensor import distribute_tensor

                mask_dtensor = distribute_tensor(
                    full_mask.to(local_grad.device),
                    grad.device_mesh,
                    grad.placements,
                )
                mask = get_local_tensor(mask_dtensor)
            else:
                # Regular tensor - create mask directly
                mask = _create_expert_mask_1d(
                    size=full_shape[0],
                    num_experts=self.num_experts,
                    num_experts_to_train=self.num_experts_to_train,
                    dtype=local_grad.dtype,
                    device=local_grad.device,
                )
                # Reshape and broadcast for 2D+ params
                if len(full_shape) > 1:
                    mask = mask.view(-1, *([1] * (len(full_shape) - 1)))
                    mask = mask.expand(full_shape)

            self._mask_cache[cache_key] = mask

        return self._mask_cache[cache_key]

    def pre_optim_step(self):
        """Zero out gradients for frozen experts before optimizer step."""
        masked_count = 0

        for name, param in self.trainer.train_module.model.named_parameters():
            if not self._should_mask(name):
                continue

            if param.grad is None:
                continue

            # Get the full shape (before any sharding)
            if isinstance(param, DTensor):
                full_shape = param.shape  # DTensor.shape gives full shape
            else:
                full_shape = param.shape

            # Get or create the mask
            mask = self._get_or_create_mask(name, param.grad, full_shape)

            # Apply mask in-place to local gradient
            local_grad = get_local_tensor(param.grad)
            local_grad.mul_(mask)
            masked_count += 1

        # Log on first step
        if not self._logged_params:
            log.info(
                f"FrozenExpertGradientMask: Masking gradients for {masked_count} parameters "
                f"(freezing {self.num_experts - self.num_experts_to_train}/{self.num_experts} experts)"
            )
            self._logged_params = True
