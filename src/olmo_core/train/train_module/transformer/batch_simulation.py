"""Utilities for opt-in smaller-batch optimization simulation."""

import math
import random
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from olmo_core.distributed.utils import get_local_tensor


def structured_noise_loss_scales(
    num_ghost_batches: int, *, seed: int, step: int
) -> tuple[float, ...]:
    """
    Return dense loss scales that make a global batch mimic a smaller batch.

    Let ``g_j`` be the gradient of ghost batch ``j`` and ``G`` be the number of ghost
    batches. Normal training produces ``mean(g_j)``. This function returns scales
    ``alpha_j`` such that the gradient produced by the existing global-batch loss is::

        mean(alpha_j * g_j) = mean(g_j) + structured_noise

    The scales obey ``sum(alpha) = G`` and ``sum(alpha ** 2) = G ** 2``. Consequently,
    when the ghost gradients are i.i.d., the resulting gradient has the same covariance
    as one ghost-batch gradient. The centered random contrast is dense, so all examples
    in the global batch contribute. It also preserves the full empirical, anisotropic
    gradient covariance rather than approximating it with isotropic Gaussian noise.

    The draw is deterministic in ``seed`` and ``step`` so every distributed rank applies
    identical weights and checkpoint resumes reproduce the same sequence.
    """
    if num_ghost_batches < 1:
        raise ValueError("'num_ghost_batches' must be at least 1")
    if num_ghost_batches == 1:
        return (1.0,)

    rng = random.Random((int(seed) << 64) ^ int(step))
    contrast = [rng.gauss(0.0, 1.0) for _ in range(num_ghost_batches)]
    contrast_mean = sum(contrast) / num_ghost_batches
    contrast = [value - contrast_mean for value in contrast]
    contrast_norm = math.sqrt(sum(value * value for value in contrast))

    # This is practically impossible for Gaussian draws, but keep the function total and
    # deterministic for mocked RNGs and unusual platforms.
    if contrast_norm == 0.0:
        contrast = [1.0, -1.0] + [0.0] * (num_ghost_batches - 2)
        contrast_norm = math.sqrt(2.0)

    target_norm = math.sqrt(num_ghost_batches * (num_ghost_batches - 1))
    scale = target_norm / contrast_norm
    return tuple(1.0 + scale * value for value in contrast)


@torch.no_grad()
def average_module_and_optimizer_state(
    module: nn.Module,
    optimizer: Optimizer,
    process_group: dist.ProcessGroup,
) -> None:
    """Average parameter shards and tensor optimizer state across local-SGD replicas."""
    replica_count = dist.get_world_size(process_group)
    if replica_count <= 1:
        return

    tensors = list(_local_parameter_tensors(module))
    tensors.extend(_local_optimizer_state_tensors(optimizer))
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
        tensor.div_(replica_count)


def _local_parameter_tensors(module: nn.Module) -> Iterable[torch.Tensor]:
    for parameter in module.parameters():
        yield get_local_tensor(parameter.detach())


def _local_optimizer_state_tensors(optimizer: Optimizer) -> Iterable[torch.Tensor]:
    seen = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                tensor = get_local_tensor(value)
                # Optimizers can theoretically alias state tensors. Do not average an alias twice.
                identity = id(tensor)
                if identity not in seen:
                    seen.add(identity)
                    yield tensor
