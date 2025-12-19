import logging
from dataclasses import dataclass
from typing import ClassVar, List, Optional

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

    :param layer_names: List of layer name patterns to monitor (e.g., ["router", "expert_0"]).
        Gradients for parameters containing any of these substrings will be monitored.
    :param max_steps_to_monitor: Maximum number of initial steps to monitor gradients.
        After this, only monitors at logging intervals. Set to None to always monitor at intervals.
    :param log_all_params: If True, logs all parameter gradients matching layer_names.
        If False, only logs summary statistics (count of zero/non-zero gradients).
    """

    priority: ClassVar[int] = -1  # Run late, after most other callbacks

    layer_names: List[str]
    max_steps_to_monitor: Optional[int] = 10
    log_all_params: bool = False

    def pre_optim_step(self):
        """Check gradients after backward, before optimizer step."""
        # Determine if we should monitor gradients this step
        should_monitor = False

        if self.max_steps_to_monitor is not None and self.step <= self.max_steps_to_monitor:
            # Always monitor for the first N steps
            should_monitor = True
        elif self.step % self.trainer.metrics_collect_interval == 0:
            # Monitor at logging intervals
            should_monitor = True

        if not should_monitor:
            return

        # Track statistics
        zero_grads_count = 0
        nonzero_grads_count = 0
        total_monitored = 0

        for name, param in self.trainer.train_module.model.named_parameters():
            # Check if this parameter matches any of the layers we want to monitor
            if not any(layer_name in name for layer_name in self.layer_names):
                continue

            total_monitored += param.numel() if param.grad is not None else param.numel()

            if param.grad is not None:
                # Check if all individual elements are zero
                num_zero_grads = param.grad.count_nonzero().item()
                zero_grads_count += num_zero_grads
                nonzero_grads_count += param.numel() - num_zero_grads

                # Optionally log individual parameter gradients
                if self.log_all_params:
                    grad_norm = param.grad.norm().item()
                    self.trainer.record_metric(
                        f"grad_norm/{name}",
                        grad_norm,
                        namespace="debug",
                        reduce_type=ReduceType.max,
                    )
            else:
                # Gradient is None (parameter not in computation graph)
                zero_grads_count += param.numel()

        # Always log summary statistics
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

            if self.step <= 5:  # Log to console for first few steps
                log.info(
                    f"Step {self.step}: Monitored {total_monitored} parameters "
                    f"({zero_grads_count} zero, {nonzero_grads_count} non-zero gradients)"
                )
