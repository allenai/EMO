from dataclasses import dataclass
from typing import List, Optional

from olmo_core.train.callbacks.callback import Callback
from olmo_core.nn.moe.twolevel_batchlb_router import MoETwoLevelBatchLBRouter

@dataclass
class ExpertPoolSchedulerCallback(Callback):
    min_pool: int          # your CLI / config document_expert_pool (minimum)
    decay_steps: int       # steps to decay from full pool -> min_pool

    def post_attach(self):
        breakpoint()
        # Called once after attaching to the trainer; cache routers here.
        model = self.trainer.train_module.model  # Transformer
        routers: List[MoETwoLevelBatchLBRouter] = []
        for m in model.modules():
            if isinstance(m, MoETwoLevelBatchLBRouter):
                routers.append(m)

        # if routers is empty, return
        if not routers or self.min_pool==-1 or self.decau_steps==-1:
            return

        self._routers = routers
        self._num_experts: Optional[int] = (
            routers[0].num_experts if routers else None
        )

    def pre_step(self, batch):
        # Runs before each training step; adjust pool based on global_step.
        breakpoint()
        del batch
        if not getattr(self, "_routers", None) or self._num_experts is None:
            return

        step = self.step  # this is trainer.global_step

        start = self._num_experts
        end = self.min_pool
        if step >= self.decay_steps:
            current_pool = end
        else:
            frac = step / float(self.decay_steps)
            pool = start - frac * (start - end)
            pool = max(end, min(start, pool))
            current_pool = int(round(pool))

        for r in self._routers:
            r.document_expert_pool = current_pool