import logging
from dataclasses import dataclass
from typing import ClassVar, List, Optional

from olmo_core.distributed.utils import get_local_tensor

from ..common import ReduceType
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GradientMonitorCallback(Callback):
    """
    Monitor gradients for specific layers to verify they are zero or non-zero as expected.

    This callback checks gradients only:
    - For the first `max_steps_to_monitor` steps (default: 10), OR
    - At logging intervals (every `trainer.metrics_collect_interval` steps)

    It tracks two levels of granularity:

    1. **Element-level** (for metrics): counts individual zero/non-zero gradient elements
       across matched parameters, and records to the trainer metric system.
    2. **Param-level** (for logs): classifies each parameter as having non-zero grad,
       all-zero grad (e.g. masked by ``FrozenExpertGradientMaskCallback``), or no grad
       (``requires_grad=False`` / grad is ``None``). Logged to console on the first
       monitored step.

    :param layer_names: List of layer name patterns to monitor (e.g., ["router", "expert_0"]).
        Gradients for parameters containing any of these substrings will be monitored.
        If empty, monitors all parameters.
    :param max_steps_to_monitor: Maximum number of initial steps to monitor gradients.
        After this, only monitors at logging intervals. Set to None to always monitor at intervals.
    :param log_all_params: If True, logs all parameter gradients matching layer_names.
        If False, only logs summary statistics (count of zero/non-zero gradients).
    """

    priority: ClassVar[int] = -1  # Run late, after most other callbacks

    layer_names: List[str]
    max_steps_to_monitor: Optional[int] = 10
    log_all_params: bool = False

    def _should_monitor(self, name: str) -> bool:
        if not self.layer_names:
            return True
        return any(layer_name in name for layer_name in self.layer_names)

    def pre_optim_step(self):
        """Check gradients after backward, before optimizer step."""
        # Determine if we should monitor gradients this step
        should_monitor = False

        if self.max_steps_to_monitor is not None and self.step <= self.max_steps_to_monitor:
            should_monitor = True
        elif self.step % self.trainer.metrics_collect_interval == 0:
            should_monitor = True

        if not should_monitor:
            return

        # Element-level counts (for metrics)
        zero_grads_count = 0
        nonzero_grads_count = 0
        total_monitored = 0

        # Param-level classification (for logging)
        no_grad_params = []      # requires_grad=False or grad is None
        zero_grad_params = []    # has grad but all zeros (e.g. masked)
        nonzero_grad_params = []  # has non-zero grad

        for name, param in self.trainer.train_module.model.named_parameters():
            if not self._should_monitor(name):
                continue

            total_monitored += param.numel()

            if not param.requires_grad or param.grad is None:
                no_grad_params.append(name)
                zero_grads_count += param.numel()
                continue

            local_grad = get_local_tensor(param.grad)
            num_nonzero = local_grad.count_nonzero().item()
            nonzero_grads_count += num_nonzero
            zero_grads_count += param.numel() - num_nonzero

            if num_nonzero == 0:
                zero_grad_params.append(name)
            else:
                nonzero_grad_params.append(name)

            if self.log_all_params:
                grad_norm = local_grad.norm().item()
                self.trainer.record_metric(
                    f"grad_norm/{name}",
                    grad_norm,
                    namespace="debug",
                    reduce_type=ReduceType.max,
                )

        # Record element-level metrics
        if total_monitored > 0:
            self.trainer.record_metric(
                "grad_monitor/zero_grads_count",
                zero_grads_count,
                namespace="debug",
            )
            self.trainer.record_metric(
                "grad_monitor/nonzero_grads_count",
                nonzero_grads_count,
                namespace="debug",
            )
            self.trainer.record_metric(
                "grad_monitor/zero_grads_pct",
                100.0 * zero_grads_count / total_monitored,
                namespace="debug",
            )

        # Log param-level breakdown on first few steps
        if self.step <= 5:
            log.info(
                f"GradientMonitor (step {self.step}):\n"
                f"  {len(nonzero_grad_params)} params with non-zero grad\n"
                f"  {len(zero_grad_params)} params with all-zero grad (masked)\n"
                f"  {len(no_grad_params)} params with no grad (requires_grad=False or None)\n"
                f"  Element counts: {nonzero_grads_count:,d} non-zero, "
                f"{zero_grads_count:,d} zero out of {total_monitored:,d} total"
            )
            if self.step <= 1:
                if zero_grad_params:
                    log.info("  All-zero grad params: %s", zero_grad_params)
                if no_grad_params:
                    log.info("  No-grad params: %s", no_grad_params)
