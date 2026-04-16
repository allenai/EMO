import torch
import torch.distributed as dist

from transformers import TrainerCallback


class LogMoeCallback(TrainerCallback):
    """
    Logs MoE losses during training.

    - forward hook: stash latest aux_loss (no counting here)
    - on_substep_end/on_step_end: accumulate once per micro-step / step
    - on_log: write averaged aux_loss into `logs` so wandb logs it
    """

    def __init__(
        self,
        reduce_across_processes=True,  # good default for DDP/FSDP
    ):
        self.lb_loss_key = "lb_loss"
        self.ce_loss_key = "ce_loss"
        self.reduce_across_processes = reduce_across_processes

        self._handle = None
        self._latest_lb_loss = None  # tensor
        self._latest_ce_loss = None  # tensor
        self._window_lb_sum = None  # tensor
        self._window_ce_sum = None  # tensor

        self._globalstep_last_logged = 0

    def _extract_aux_loss(self, output, aux_loss_key):
        aux = getattr(output, aux_loss_key, None)
        if aux is not None:
            return aux
        if isinstance(output, dict):
            return output.get(aux_loss_key, None)
        return None

    def _forward_hook(self, module, inputs, output):
        lb_loss = self._extract_aux_loss(output, self.lb_loss_key)
        ce_loss = self._extract_aux_loss(output, self.ce_loss_key)

        # add on model's router_aux_loss_coef scaling since that is done in the trainer
        router_aux_loss_coef = module.config.router_aux_loss_coef
        lb_loss = lb_loss * router_aux_loss_coef
        ce_loss = ce_loss

        # stash *latest* value only (avoid checkpoint recompute double counting)
        if lb_loss is not None:
            self._latest_lb_loss = lb_loss.detach()
        if ce_loss is not None:
            self._latest_ce_loss = ce_loss.detach()
        return

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        m = model.module if hasattr(model, "module") else model
        self._handle = m.register_forward_hook(self._forward_hook)
        self._window_lb_sum = torch.tensor(0.0, device=args.device)
        self._window_ce_sum = torch.tensor(0.0, device=args.device)

    def on_train_end(self, args, state, control, **kwargs):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._latest_lb_loss = None
        self._latest_ce_loss = None
        self._window_lb_sum = None
        self._window_ce_sum = None

    def _accumulate_latest(self):
        if self._latest_lb_loss is not None:
            self._window_lb_sum += self._latest_lb_loss.float().to(self._window_lb_sum.device)
            self._latest_lb_loss = None

        if self._latest_ce_loss is not None:
            self._window_ce_sum += self._latest_ce_loss.float().to(self._window_ce_sum.device)
            self._latest_ce_loss = None

    def on_substep_end(self, args, state, control, **kwargs):
        # called on non-sync micro-steps (gradient accumulation)
        self._accumulate_latest()

    def on_step_end(self, args, state, control, **kwargs):
        # called on optimizer update steps (the last micro-step)
        self._accumulate_latest()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step <= self._globalstep_last_logged:
            return

        use_dist = (
            self.reduce_across_processes
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

        # --- LB ---
        lb = self._window_lb_sum
        if use_dist:
            # clone so we can reset the window without affecting the reduced value
            lb_reduced = lb.detach().clone()
            dist.reduce(
                lb_reduced, dst=0, op=dist.ReduceOp.SUM
            )  # reduce by summing across DP rank (since we have already divided by the global batch size
        else:
            lb_reduced = lb.detach().clone()

        # --- CE ---

        ce = self._window_ce_sum
        if use_dist:
            ce_reduced = ce.detach().clone()
            dist.reduce(
                ce_reduced, dst=0, op=dist.ReduceOp.SUM
            )  # reduce by summing across DP ranks
        else:
            ce_reduced = ce.detach().clone()

        # Reset windows on *all* ranks so everyone stays in sync
        self._window_lb_sum.zero_()
        self._window_ce_sum.zero_()

        # Update step marker on *all* ranks (keeps denom consistent everywhere)
        steps_since = state.global_step - self._globalstep_last_logged
        self._globalstep_last_logged = state.global_step

        # Only rank 0 writes logs (and only if logs dict exists)
        if state.is_world_process_zero and logs is not None:
            logs[f"train/{self.lb_loss_key}"] = lb_reduced.item() / max(1, steps_since)
            logs[f"train/{self.ce_loss_key}"] = ce_reduced.item() / max(1, steps_since)
        return
