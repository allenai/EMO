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


def clone_local_parameter_tensors(module: nn.Module) -> list[torch.Tensor]:
    """Clone the local parameter shards used as the start of a DiLoCo outer round."""
    return [
        get_local_tensor(parameter.detach()).clone()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


@torch.no_grad()
def recalibrate_adam_second_moment_for_batch_size(
    optimizer: Optimizer,
    *,
    batch_size_ratio: float,
) -> int:
    """Adjust loaded Adam second moments for a smaller effective batch.

    For a stochastic gradient ``g``, ``E[g^2] = E[g]^2 + Var(g)``.  Switching from a
    conventional global batch to a smaller local batch should leave the signal term alone while
    increasing the variance term by the batch-size ratio.  We estimate the two terms from Adam's
    bias-corrected first and second moments, clamp the estimated variance at zero, and write the
    adjusted value back in Adam's original biased representation.

    The function deliberately leaves the first moment and step counter unchanged.  It returns the
    number of parameter states adjusted and raises if the optimizer is not an initialized Adam-like
    optimizer, preventing a requested recalibration from silently doing nothing.
    """
    if not math.isfinite(batch_size_ratio) or batch_size_ratio < 1.0:
        raise ValueError("'batch_size_ratio' must be finite and at least 1")

    adjusted = 0
    for group in optimizer.param_groups:
        betas = group.get("betas")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise RuntimeError("second-moment recalibration requires Adam-style 'betas'")
        beta1, beta2 = (float(value) for value in betas)
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            first_moment = state.get("exp_avg")
            second_moment = state.get("exp_avg_sq")
            step = state.get("step")
            if not all(isinstance(value, torch.Tensor) for value in (first_moment, second_moment)):
                raise RuntimeError(
                    "second-moment recalibration requires initialized Adam exp_avg/exp_avg_sq state"
                )
            local_first = get_local_tensor(first_moment)
            local_second = get_local_tensor(second_moment)
            local_step = get_local_tensor(step) if isinstance(step, torch.Tensor) else step
            if isinstance(local_step, torch.Tensor):
                if local_step.numel() != 1:
                    raise RuntimeError("Adam step state must be scalar")
                step_value = float(local_step.item())
            elif isinstance(local_step, (int, float)):
                step_value = float(local_step)
            else:
                raise TypeError("second-moment recalibration requires initialized Adam step state")
            if step_value <= 0:
                raise RuntimeError("Adam step must be positive for second-moment recalibration")

            bias_correction1 = 1.0 - beta1**step_value
            bias_correction2 = 1.0 - beta2**step_value
            # Use one temporary tensor per parameter.  Mutating v in place avoids materializing
            # multiple full-size optimizer tensors for large embedding matrices.
            # Keep ``exp_avg`` bit-for-bit unchanged. ``Tensor.float()`` may alias an already
            # float32 tensor, so the square must be out-of-place.
            signal_squared = local_first.detach().float().square().div_(bias_correction1**2)
            original_dtype = local_second.dtype
            if original_dtype != torch.float32:
                corrected_second = local_second.detach().float().div_(bias_correction2)
                corrected_second.sub_(signal_squared).clamp_min_(0.0)
                corrected_second.mul_(batch_size_ratio).add_(signal_squared).mul_(bias_correction2)
                local_second.copy_(corrected_second.to(dtype=original_dtype))
            else:
                local_second.div_(bias_correction2)
                local_second.sub_(signal_squared).clamp_min_(0.0)
                local_second.mul_(batch_size_ratio).add_(signal_squared).mul_(bias_correction2)
            adjusted += 1

    if adjusted == 0:
        raise RuntimeError("second-moment recalibration found no initialized Adam parameter state")
    return adjusted


@torch.no_grad()
def diloco_outer_step(
    module: nn.Module,
    outer_optimizer: Optimizer,
    outer_parameters: list[torch.Tensor],
    process_group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    """Apply one synchronous DiLoCo outer step and reset every replica to its result.

    Each replica starts an outer round from ``outer_parameters`` and independently updates
    ``module`` with its inner optimizer. The pseudo-gradient is the start parameter minus the
    local parameter. Pseudo-gradients are averaged across replicas, then applied to the shared
    start parameters by ``outer_optimizer``. The inner optimizer is deliberately untouched so
    every replica retains its own AdamW moments across outer rounds, as in DiLoCo.
    """
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if len(parameters) != len(outer_parameters):
        raise RuntimeError(
            "DiLoCo outer parameter snapshot does not match the module parameter count"
        )

    replica_count = dist.get_world_size(process_group)
    outer_optimizer.zero_grad(set_to_none=True)
    displacement_sq = torch.zeros((), device=outer_parameters[0].device, dtype=torch.float32)
    dispersion_sq = torch.zeros_like(displacement_sq)
    momentum_sq = torch.zeros_like(displacement_sq)
    momentum_dot_displacement = torch.zeros_like(displacement_sq)
    for parameter, outer_parameter in zip(parameters, outer_parameters):
        local_parameter = get_local_tensor(parameter.detach())
        if local_parameter.shape != outer_parameter.shape:
            raise RuntimeError("DiLoCo outer parameter snapshot has an incompatible shape")

        # Preserve the parameter's distributed layout in the synthetic gradient so the regular
        # PyTorch optimizer can update DTensor/FSDP parameters without special-casing them.
        pseudo_gradient = parameter.detach().clone()
        local_pseudo_gradient = get_local_tensor(pseudo_gradient)
        local_pseudo_gradient.copy_(outer_parameter).sub_(local_parameter)
        if replica_count > 1:
            dist.all_reduce(
                local_pseudo_gradient,
                op=dist.ReduceOp.SUM,
                group=process_group,
            )
            local_pseudo_gradient.div_(replica_count)

        local_endpoint_mean = outer_parameter - local_pseudo_gradient
        local_deviation = local_parameter - local_endpoint_mean
        dispersion_sq.add_(local_deviation.float().square().sum())
        displacement_sq.add_(local_pseudo_gradient.float().square().sum())
        momentum_buffer = outer_optimizer.state.get(parameter, {}).get("momentum_buffer")
        if momentum_buffer is not None:
            local_momentum = get_local_tensor(momentum_buffer.detach())
            momentum_sq.add_(local_momentum.float().square().sum())
            momentum_dot_displacement.add_(
                (local_momentum.float() * local_pseudo_gradient.float()).sum()
            )

        # The outer optimizer acts on the shared parameter at the beginning of the round, not
        # on any particular worker's local endpoint.
        local_parameter.copy_(outer_parameter)
        parameter.grad = pseudo_gradient

    outer_optimizer.step()
    outer_optimizer.zero_grad(set_to_none=True)

    update_sq = torch.zeros_like(displacement_sq)
    for parameter, outer_parameter in zip(parameters, outer_parameters):
        updated_parameter = get_local_tensor(parameter.detach())
        update_sq.add_((updated_parameter.float() - outer_parameter.float()).square().sum())
        outer_parameter.copy_(updated_parameter)

    # The default group spans replica and shard dimensions. Replica-reduced displacement and
    # update values are duplicated once per replica, so divide them after the global reduction.
    telemetry = torch.stack(
        [
            displacement_sq,
            update_sq,
            momentum_sq,
            momentum_dot_displacement,
            dispersion_sq,
        ]
    )
    if dist.is_initialized():
        dist.all_reduce(telemetry, op=dist.ReduceOp.SUM)
        telemetry.div_(replica_count)
    displacement_norm = telemetry[0].sqrt()
    update_norm = telemetry[1].sqrt()
    momentum_norm = telemetry[2].sqrt()
    cosine = telemetry[3] / (momentum_norm * displacement_norm).clamp_min(1e-30)
    return {
        "averaged local displacement norm": displacement_norm,
        "outer update norm": update_norm,
        "outer update / displacement": update_norm / displacement_norm.clamp_min(1e-30),
        "outer momentum norm": momentum_norm,
        "momentum / displacement cosine": cosine,
        "replica dispersion norm": telemetry[4].sqrt(),
    }


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
