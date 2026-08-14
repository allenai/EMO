import logging
from dataclasses import dataclass
from typing import Any, Dict

from olmo_core.train.train_module.transformer.batch_simulation import (
    recalibrate_adam_second_moment_for_batch_size,
)
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class AdamSecondMomentBatchRecalibrationCallback(Callback):
    """Recalibrate loaded Adam variance once when reducing the real global batch size."""

    batch_size_ratio: float
    expected_step: int
    _applied: bool = False

    def state_dict(self) -> Dict[str, Any]:
        return {"applied": self._applied}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._applied = bool(state_dict.get("applied", False))

    def pre_train(self):
        if self._applied:
            log.info("Adam second-moment batch recalibration was already applied; skipping")
            return
        if self.trainer.global_step != self.expected_step:
            raise RuntimeError(
                "Adam second-moment batch recalibration expected checkpoint step "
                f"{self.expected_step}, got {self.trainer.global_step}"
            )
        train_module = self.trainer.train_module
        if not isinstance(train_module, TransformerTrainModule):
            raise TypeError("batch recalibration requires TransformerTrainModule")
        adjusted = recalibrate_adam_second_moment_for_batch_size(
            train_module.optim,
            batch_size_ratio=self.batch_size_ratio,
        )
        self._applied = True
        log.info(
            "Recalibrated Adam second moments for %d parameter states at real batch-size "
            "ratio %.3f; first moments and optimizer step counters were unchanged",
            adjusted,
            self.batch_size_ratio,
        )
