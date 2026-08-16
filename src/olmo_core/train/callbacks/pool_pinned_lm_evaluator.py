"""
LM (perplexity) evaluator that pins the randpool router's ``eval_document_expert_pool`` for the
duration of the eval, then restores it. Attaching two of these — one with ``eval_pool=None``
(model default, i.e. full pool in the meta_learning recipes) and one with e.g. ``eval_pool=32`` —
gives the *selective-vs-full CE gap* on the same validation mix, the headline quantity of the
meta_learning experiment. Works for any arm, including the vanilla baseline.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter,
)

from .callback import Callback
from .evaluator_callback import EvaluatorCallback, LMEvaluatorCallbackConfig

if TYPE_CHECKING:
    from ..trainer import Trainer

log = logging.getLogger(__name__)


@dataclass
class PoolPinnedEvaluatorCallback(EvaluatorCallback):
    """
    :class:`EvaluatorCallback` that sets ``eval_document_expert_pool = eval_pool`` on every
    randpool router around ``perform_eval`` (and restores the previous values afterwards).
    ``eval_pool=None`` leaves the routers untouched.
    """

    eval_pool: Optional[int] = None

    def _routers(self) -> List[MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter]:
        model = self.trainer.train_module.model
        return [
            m
            for m in model.modules()
            if isinstance(m, MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouter)
        ]

    def perform_eval(self, prefix: str = "eval"):
        if self.eval_pool is None:
            return super().perform_eval(prefix=prefix)

        routers = self._routers()
        saved = [r.eval_document_expert_pool for r in routers]
        for r in routers:
            r.eval_document_expert_pool = self.eval_pool
        log.info(f"Pinned eval_document_expert_pool={self.eval_pool} on {len(routers)} routers")
        try:
            return super().perform_eval(prefix=prefix)
        finally:
            for r, old in zip(routers, saved):
                r.eval_document_expert_pool = old


@dataclass
class PoolPinnedLMEvaluatorCallbackConfig(LMEvaluatorCallbackConfig):
    """
    :class:`LMEvaluatorCallbackConfig` variant that builds a :class:`PoolPinnedEvaluatorCallback`.
    Give each instance a distinct ``name`` (e.g. ``lm-full`` / ``lm-pool32``) so the metrics land
    under distinct paths.
    """

    eval_pool: Optional[int] = None

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        cb = super().build(trainer)
        if cb is None:
            return None
        assert isinstance(cb, EvaluatorCallback)
        return PoolPinnedEvaluatorCallback(
            evaluators=cb.evaluators,
            eval_interval=cb.eval_interval,
            fixed_steps=cb.fixed_steps,
            eval_on_startup=cb.eval_on_startup,
            eval_on_finish=cb.eval_on_finish,
            cancel_after_first_eval=cb.cancel_after_first_eval,
            eval_duration=cb.eval_duration,
            log_interval=cb.log_interval,
            eval_pool=self.eval_pool,
        )
