import logging
import math
from dataclasses import dataclass
from typing import Any, Dict

from torch.optim import Optimizer

from olmo_core.optim.config import INITIAL_LR_FIELD
from olmo_core.train.train_module.transformer.batch_simulation import (
    recalibrate_adam_second_moment_for_batch_size,
)
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule

from .callback import Callback

log = logging.getLogger(__name__)


def transition_optimizer_hyperparameters(
    optim: Optimizer,
    *,
    source_lr: float,
    target_lr: float,
    source_weight_decay: float,
    target_weight_decay: float,
) -> int:
    """Validate loaded source hyperparameters and install a new LR/WD coordinate.

    Adam moments and step counters are intentionally untouched. Parameter groups with zero
    weight decay remain unregularized; every group using ``source_weight_decay`` transitions
    to ``target_weight_decay``.
    """

    transitioned_weight_decay_groups = 0
    for group_idx, group in enumerate(optim.param_groups):
        loaded_lr = float(group["lr"])
        if not math.isclose(loaded_lr, source_lr, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"optimizer group {group_idx} loaded LR {loaded_lr} != expected source LR "
                f"{source_lr}"
            )
        loaded_weight_decay = float(group.get("weight_decay", 0.0))
        if math.isclose(
            loaded_weight_decay, source_weight_decay, rel_tol=0.0, abs_tol=1e-12
        ):
            group["weight_decay"] = target_weight_decay
            transitioned_weight_decay_groups += 1
        elif not math.isclose(loaded_weight_decay, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"optimizer group {group_idx} loaded weight decay {loaded_weight_decay} is "
                f"neither zero nor expected source weight decay {source_weight_decay}"
            )
        group["lr"] = target_lr
        group[INITIAL_LR_FIELD] = target_lr
    if transitioned_weight_decay_groups == 0:
        raise RuntimeError(
            "optimizer hyperparameter transition found no source weight-decay parameter group"
        )
    return transitioned_weight_decay_groups


@dataclass
class OptimizerHyperparameterTransitionCallback(Callback):
    """Apply an intentional once-only LR/WD transition after explicit checkpoint loading."""

    expected_step: int
    source_lr: float
    target_lr: float
    source_weight_decay: float
    target_weight_decay: float
    _applied: bool = False

    def state_dict(self) -> Dict[str, Any]:
        return {"applied": self._applied}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._applied = bool(state_dict.get("applied", False))

    def pre_train(self):
        if self._applied:
            log.info("Optimizer hyperparameter transition was already applied; skipping")
            return
        if self.trainer.global_step != self.expected_step:
            raise RuntimeError(
                "Optimizer hyperparameter transition expected checkpoint step "
                f"{self.expected_step}, got {self.trainer.global_step}"
            )
        train_module = self.trainer.train_module
        if not isinstance(train_module, TransformerTrainModule):
            raise TypeError("optimizer hyperparameter transition requires TransformerTrainModule")
        adjusted = transition_optimizer_hyperparameters(
            train_module.optim,
            source_lr=self.source_lr,
            target_lr=self.target_lr,
            source_weight_decay=self.source_weight_decay,
            target_weight_decay=self.target_weight_decay,
        )
        self._applied = True
        log.info(
            "Transitioned optimizer hyperparameters at step %d: LR %.8g -> %.8g; WD %.8g "
            "-> %.8g across %d regularized groups; Adam moments and step counters unchanged",
            self.expected_step,
            self.source_lr,
            self.target_lr,
            self.source_weight_decay,
            self.target_weight_decay,
            adjusted,
        )


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
