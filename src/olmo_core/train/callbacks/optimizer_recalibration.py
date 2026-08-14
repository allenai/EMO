import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from olmo_core.optim.config import INITIAL_LR_FIELD, LR_FIELD
from olmo_core.train.train_module.transformer.batch_simulation import (
    recalibrate_adam_second_moment_for_batch_size,
)
from olmo_core.train.train_module.transformer.train_module import (
    TransformerTrainModule,
    assert_optimizer_hyperparameters_match,
    optimizer_hyperparameters,
)

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class AdamSecondMomentBatchRecalibrationCallback(Callback):
    """Recalibrate loaded Adam variance once when reducing the real global batch size."""

    batch_size_ratio: float
    expected_step: int
    restart_lr: Optional[float] = None
    _applied: bool = False
    _expected_optimizer_hyperparameters: Optional[List[Dict[str, Any]]] = field(
        default=None, init=False, repr=False
    )

    def state_dict(self) -> Dict[str, Any]:
        return {"applied": self._applied}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._applied = bool(state_dict.get("applied", False))

    def pre_checkpoint_loaded(self, path):
        del path
        if self.restart_lr is None:
            return
        train_module = self.trainer.train_module
        if not isinstance(train_module, TransformerTrainModule):
            raise TypeError("optimizer LR restart requires TransformerTrainModule")
        self._expected_optimizer_hyperparameters = optimizer_hyperparameters(train_module.optim)
        for group_idx, group in enumerate(self._expected_optimizer_hyperparameters):
            if group.get(LR_FIELD) != self.restart_lr or group.get(INITIAL_LR_FIELD) != self.restart_lr:
                raise RuntimeError(
                    f"optimizer group {group_idx} was configured with LR {group.get(LR_FIELD)} "
                    f"and initial LR {group.get(INITIAL_LR_FIELD)}, expected {self.restart_lr}"
                )

    def post_checkpoint_loaded(self, path):
        del path
        if self.restart_lr is None:
            return
        train_module = self.trainer.train_module
        if not isinstance(train_module, TransformerTrainModule):
            raise TypeError("optimizer LR restart requires TransformerTrainModule")
        if self._expected_optimizer_hyperparameters is None:
            raise RuntimeError("optimizer hyperparameters were not captured before checkpoint load")
        loaded = optimizer_hyperparameters(train_module.optim)
        for group_idx, (expected_group, loaded_group, group) in enumerate(
            zip(
                self._expected_optimizer_hyperparameters,
                loaded,
                train_module.optim.param_groups,
                strict=True,
            )
        ):
            unexpected = {
                key: (expected_group.get(key), loaded_group.get(key))
                for key in set(expected_group) | set(loaded_group)
                if key not in {LR_FIELD, INITIAL_LR_FIELD}
                and expected_group.get(key) != loaded_group.get(key)
            }
            if unexpected:
                raise RuntimeError(
                    f"optimizer group {group_idx} has non-LR checkpoint hyperparameter "
                    f"mismatches: {unexpected}"
                )
            group[LR_FIELD] = expected_group[LR_FIELD]
            group[INITIAL_LR_FIELD] = expected_group[INITIAL_LR_FIELD]
        assert_optimizer_hyperparameters_match(
            self._expected_optimizer_hyperparameters,
            optimizer_hyperparameters(train_module.optim),
        )
        log.info(
            "Restored configured optimizer LR %.8f after loading the post-decay checkpoint; "
            "all non-LR optimizer hyperparameters matched exactly",
            self.restart_lr,
        )

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
