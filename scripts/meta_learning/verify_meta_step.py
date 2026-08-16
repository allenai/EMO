# PARENT: none (new verification harness for the meta_learning experiment)
# DESCRIPTION:
#     Mechanism-correctness gate for MetaLearningTransformerTrainModule. Must pass before any
#     Beaker launch. Builds a tiny randpool EMO model (fp32, compile off, LB/z losses off so the
#     loss is pure CE) and checks, on a synthetic multi-document batch:
#
#       0. Determinism floor: two identical alpha=0 runs -> |dL| calibrates all tolerances.
#       B. Oracle: `same_tokens, inner_lr=0` == `outer_only` (outer CE and final grads match)
#          -> validates the stash/zero/restore plumbing is a no-op at alpha=0.
#       1. Perturbation visibility: outer CE at alpha=eps differs from alpha=0 well beyond the
#          determinism floor -> the in-place pseudo-step actually reaches the outer forward.
#       2. Directional derivative: (L(0) - L(eps))/eps ~= <g_inner, g_outer(theta)> (expert
#          params) -> the pseudo-step has the right sign, scale, and row-targeting.
#       3. Manual reference: replicate the whole step by hand (autograd g_inner -> explicit
#          theta' -> autograd g_outer(theta')) and require the module's final .grad to equal
#          lambda*g_inner + g_outer(theta'), at lambda=0 and lambda=0.5.
#       H. heldout mode smoke: runs, finite losses, weights restored.
#
#     EMO_META_CHECK_RESTORE=1 is set for every run (bitwise weight-restore assert inside the
#     module), and the script independently verifies expert weights are unchanged after each step.
#
#   PYTHONPATH=src torchrun --nproc-per-node=1 scripts/meta_learning/verify_meta_step.py
#   PYTHONPATH=src torchrun --nproc-per-node=2 scripts/meta_learning/verify_meta_step.py
##############################################################

import logging
import os
import sys

os.environ.setdefault("EMO_META_CHECK_RESTORE", "1")

import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    MetaLearningTransformerTrainModule,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.utils import get_default_device

log = logging.getLogger("verify_meta_step")

SEQ_LEN = 512
NUM_INSTANCES = 4  # per rank; rank_microbatch = 2 instances -> 2 micro-batches (heldout-safe)
NUM_EXPERTS = 16
NUM_SHARED = 1
INNER_POOL = 4


def build_module() -> MetaLearningTransformerTrainModule:
    tokenizer = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmoe_1B_7B(
        vocab_size=tokenizer.padded_vocab_size(),
        d_model=256,
        n_layers=2,
        n_heads=4,
        num_experts=NUM_EXPERTS,
    )
    moe = model_config.block.feed_forward_moe
    assert moe is not None
    # Pure-CE loss for clean finite-difference math.
    moe.lb_loss_weight = None
    moe.z_loss_weight = None
    router_kwargs = moe.router.as_dict(exclude_none=True, recurse=False)
    router_kwargs.pop("name")
    router_kwargs.update(
        min_document_expert_pool=INNER_POOL,
        max_document_expert_pool=NUM_EXPERTS,
        eval_document_expert_pool=NUM_EXPERTS,
        eos_token_id=tokenizer.eos_token_id,
        num_shared_experts=NUM_SHARED,
    )
    moe.router = MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig(**router_kwargs)

    model = model_config.build(init_device="meta")

    dp_config = None
    if is_distributed():
        # fp32 all the way so the pseudo-step perturbation isn't quantized by a bf16 all-gather.
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.float32,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        )

    return MetaLearningTransformerTrainModule(
        model=model,
        optim=AdamWConfig(lr=1e-3, fused=torch.cuda.is_available()),
        rank_microbatch_size=2 * SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        compile_model=False,
        dp_config=dp_config,
        z_loss_multiplier=None,  # pure CE
        max_grad_norm=1.0,
        meta_mode="same_tokens",
        inner_lr=0.0,
        inner_pool_size=INNER_POOL,
        lambda_inner=0.0,
        lb_on_inner=False,
        inner_grad_clip=None,  # keep the manual reference math exact
        log_grad_cosine=True,
    )


def patch_metric_capture(module):
    captured = {}

    def record_metric(name, value, reduce_type=None, namespace=None, **kwargs):
        key = f"{namespace}/{name}" if namespace else name
        captured[key] = (
            float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        )

    def record_ce_loss(value, reduce_type=None):
        record_metric("ce", value)

    module.record_metric = record_metric
    module.record_ce_loss = record_ce_loss
    return captured


def make_batch(device) -> dict:
    """Synthetic rank batch: random tokens with EOS every ~160 tokens so seg_id sees many docs.
    Every rank uses a different seed so DP reduction is exercised with distinct data."""
    rank = dist.get_rank() if is_distributed() else 0
    g = torch.Generator().manual_seed(1234 + rank)
    tokenizer = TokenizerConfig.dolma2()
    ids = torch.randint(0, tokenizer.padded_vocab_size(), (NUM_INSTANCES, SEQ_LEN), generator=g)
    ids[:, 80::160] = tokenizer.eos_token_id
    return {"input_ids": ids.to(device)}


def mean_over_ranks(value: float, device) -> float:
    if not is_distributed():
        return value
    t = torch.tensor(value, device=device)
    dist.all_reduce(t)
    return (t / dist.get_world_size()).item()


def expert_weight_snapshot(module):
    return {p: get_local_tensor(p.data).clone() for p in module._expert_params}


def assert_weights_unchanged(module, snapshot, label):
    for p, w in snapshot.items():
        assert torch.equal(get_local_tensor(p.data), w), f"[{label}] expert weights changed!"


def run_step(module, base_batch, device, *, meta_mode, inner_lr, lambda_inner):
    """One train_batch under the given knobs. Returns (captured_metrics, final_grads_by_name)."""
    module.meta_mode = meta_mode
    module.inner_lr = inner_lr
    module.lambda_inner = lambda_inner
    captured = patch_metric_capture(module)
    module.optim.zero_grad(set_to_none=True)
    snapshot = expert_weight_snapshot(module)
    module.train_batch({"input_ids": base_batch["input_ids"].clone()})
    assert_weights_unchanged(module, snapshot, f"{meta_mode},a={inner_lr},l={lambda_inner}")
    grads = {
        name: get_local_tensor(p.grad).clone()
        for name, p in module.model.named_parameters()
        if p.grad is not None
    }
    captured["ce_global"] = mean_over_ranks(captured["ce"], device)
    return captured, grads


def manual_reference(module, base_batch, device, *, inner_lr, lambda_inner):
    """Hand-rolled same_tokens FOMAML step. Returns (outer_ce_global, expected_grads_by_name)."""
    batch = {"input_ids": base_batch["input_ids"].clone()}
    module._set_model_mode("train")
    batch["labels"] = get_labels(batch, label_ignore_index=module.label_ignore_index)
    micro_batches = split_batch(batch, module.rank_microbatch_size // SEQ_LEN)
    div, _, _ = module._phase_loss_div(micro_batches)
    prepared = [module._prepare_batch(mb) for mb in micro_batches]

    def phase(force_pool, skip_aux, z_mult, div_factor):
        ce_total = torch.tensor(0.0, device=device)
        module._set_router_meta_state(force_pool=force_pool, skip_aux=skip_aux)
        for i, (input_ids, labels, model_kwargs) in enumerate(prepared):
            with module._train_microbatch_context(i, len(prepared)):
                _, loss, ce_loss, _ = module.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=module.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=z_mult,
                    loss_div_factor=div_factor,
                    return_logits=False,
                    **model_kwargs,
                )
                ce_total += get_local_tensor(ce_loss.detach())
                loss.backward()
        return ce_total

    # Inner pass -> g_inner for every param.
    module.optim.zero_grad(set_to_none=True)
    phase(module.inner_pool_size, not module.lb_on_inner, None, div)
    g_inner = {
        name: get_local_tensor(p.grad).clone()
        for name, p in module.model.named_parameters()
        if p.grad is not None
    }
    named_expert = {
        name: p
        for name, p in module.model.named_parameters()
        if any(p is q for q in module._expert_params)
    }

    # theta' on expert weights (exact restore via clones).
    w_saved = {name: get_local_tensor(p.data).clone() for name, p in named_expert.items()}
    for name, p in named_expert.items():
        get_local_tensor(p.data).add_(g_inner[name], alpha=-inner_lr)

    # Outer pass at theta' -> g_outer.
    module.optim.zero_grad(set_to_none=True)
    outer_ce = phase(module._num_nonshared_experts, False, module.z_loss_multiplier, div)
    g_outer = {
        name: get_local_tensor(p.grad).clone()
        for name, p in module.model.named_parameters()
        if p.grad is not None
    }

    # Restore + reset router state.
    for name, p in named_expert.items():
        get_local_tensor(p.data).copy_(w_saved[name])
    module._set_router_meta_state(force_pool=None, skip_aux=False)
    module.optim.zero_grad(set_to_none=True)

    expected = {}
    for name, go in g_outer.items():
        expected[name] = go + lambda_inner * g_inner.get(name, torch.zeros_like(go))
    return mean_over_ranks(outer_ce.item(), device), expected


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def max_grad_rel_diff(got: dict, want: dict) -> float:
    worst = 0.0
    for name, w in want.items():
        g = got.get(name)
        assert g is not None, f"missing grad for {name}"
        denom = w.norm().item() + 1e-12
        worst = max(worst, (g - w).norm().item() / denom)
    return worst


def main():
    prepare_training_environment(seed=0)
    device = get_default_device()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []

    try:
        module = build_module()
        base_batch = make_batch(device)

        # --- Check 0: determinism floor ---
        cap_a, grads_a = run_step(
            module, base_batch, device, meta_mode="same_tokens", inner_lr=0.0, lambda_inner=0.0
        )
        cap_b, _ = run_step(
            module, base_batch, device, meta_mode="same_tokens", inner_lr=0.0, lambda_inner=0.0
        )
        det_noise = abs(cap_a["ce_global"] - cap_b["ce_global"])
        results.append(("0 determinism floor", True, f"|dL|={det_noise:.3e}"))
        L0 = cap_a["ce_global"]
        dot0 = cap_a["train/meta inner-outer grad dot (experts)"]
        g_norm = cap_a["train/meta inner grad norm (experts)"]

        # --- Check B: alpha=0 same_tokens == outer_only ---
        cap_o, grads_o = run_step(
            module, base_batch, device, meta_mode="outer_only", inner_lr=0.0, lambda_inner=0.0
        )
        d_loss = abs(L0 - cap_o["ce_global"])
        d_grad = max_grad_rel_diff(grads_a, grads_o)
        tol = max(10 * det_noise, 1e-6)
        ok = d_loss <= tol and d_grad <= 1e-4
        results.append(
            (
                "B alpha=0 == outer_only",
                ok,
                f"|dL|={d_loss:.3e} (tol {tol:.1e}), grad rel diff={d_grad:.3e}",
            )
        )

        # --- Checks 1+2: perturbation visibility & directional derivative ---
        w_sq = module._expert_global_sq_norm(
            [get_local_tensor(p.data) for p in module._expert_params]
        )
        base_eps = 1e-4 * float(w_sq.sqrt()) / max(g_norm, 1e-12)
        vis_ok, fd_best, fd_eps = False, float("inf"), None
        for mult in (0.1, 0.3, 1.0, 3.0):
            eps = base_eps * mult
            cap_e, _ = run_step(
                module, base_batch, device, meta_mode="same_tokens", inner_lr=eps, lambda_inner=0.0
            )
            dL = L0 - cap_e["ce_global"]
            if abs(dL) > 20 * max(det_noise, 1e-12):
                vis_ok = True
            fd = dL / eps
            err = rel_err(fd, dot0)
            if err < fd_best:
                fd_best, fd_eps = err, eps
        results.append(("1 perturbation visibility", vis_ok, f"base_eps={base_eps:.3e}"))
        results.append(
            (
                "2 directional derivative",
                fd_best < 0.10,
                f"best rel err={fd_best:.3%} at eps={fd_eps:.3e} (dot={dot0:.4e})",
            )
        )

        # --- Check 3: manual reference at lambda=0 and lambda=0.5 ---
        alpha = base_eps  # small enough for exact restore, large enough for a real perturbation
        for lam in (0.0, 0.5):
            L_ref, want = manual_reference(
                module, base_batch, device, inner_lr=alpha, lambda_inner=lam
            )
            cap_m, got = run_step(
                module,
                base_batch,
                device,
                meta_mode="same_tokens",
                inner_lr=alpha,
                lambda_inner=lam,
            )
            d = max_grad_rel_diff(got, want)
            dl = abs(cap_m["ce_global"] - L_ref)
            ok = d < 1e-3 and dl <= max(10 * det_noise, 1e-6)
            results.append(
                (f"3 manual reference (lambda={lam})", ok, f"grad rel diff={d:.3e}, |dL|={dl:.3e}")
            )

        # --- Check H: heldout smoke ---
        cap_h, _ = run_step(
            module, base_batch, device, meta_mode="heldout", inner_lr=alpha, lambda_inner=0.0
        )
        ok = all(
            cap_h.get(k) is not None and torch.isfinite(torch.tensor(cap_h[k]))
            for k in ("ce_global", "train/meta inner CE loss")
        )
        results.append(
            (
                "H heldout smoke",
                ok,
                f"outer CE={cap_h['ce_global']:.4f}, inner CE={cap_h['train/meta inner CE loss']:.4f}",
            )
        )

    finally:
        rank0 = (not is_distributed()) or dist.get_rank() == 0
        n_fail = sum(1 for _, ok, _ in results if not ok)
        if rank0:
            print("\n===== verify_meta_step results =====")
            for name, ok, detail in results:
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            print(f"===== {len(results) - n_fail}/{len(results)} passed =====\n")
        teardown_training_environment()
        if n_fail:
            sys.exit(1)


if __name__ == "__main__":
    main()
