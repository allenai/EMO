"""
Numerical check for grouped_gemm beyond 512 groups (the sparse_experts 1024-expert arms).

grouped_gemm's CUTLASS backend only accepts CUDA-resident ``batch_sizes`` for <=512 groups
(``kMaxExperts``); ``DroplessMoEMLP._gmm_batch_sizes`` moves the vector to host beyond that.
This script runs on a GPU with grouped_gemm installed (the Beaker image; the local ``emo`` env
has no grouped_gemm) and verifies, in bf16 with fp32 comparison:

  1. raw ``grouped_gemm.ops.gmm`` with HOST batch_sizes, 1024 groups (incl. empty groups),
     trans_b both ways: forward + grads wrt x and w vs a per-group ``torch.matmul`` loop.
  2. 512 groups: HOST vs DEVICE batch_sizes give identical forward/backward (same kernel,
     different descriptor build).
  3. ``DroplessMoEMLP.forward`` end to end, 1024 experts: grouped_gemm path (as in training)
     vs the module's own python-loop fallback (``_gmm = None``), forward + grads.

Exits non-zero on any mismatch so a Beaker job shows ``failed``. Extra CLI args (run name,
--data-root from launch_common's ``launch``) are ignored.
"""

import argparse
import sys

import torch

import grouped_gemm  # type: ignore  # noqa: E402  (must exist on the image)
from olmo_core.nn.moe.mlp import DroplessMoEMLP

FAILS = []


def close(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> None:
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item()
    scale = b.abs().max().item()
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f"  {'OK ' if ok else 'BAD'} {name:48s} max|diff|={err:.3e}  max|ref|={scale:.3e}")
    if not ok:
        FAILS.append(name)


def loop_gmm(x, w, batch_sizes, trans_b):
    out, start = [], 0
    for i, size in enumerate(batch_sizes.tolist()):
        rhs = w[i].t() if trans_b else w[i]
        out.append(x[start : start + size] @ rhs)
        start += size
    return torch.cat(out)


def random_batch_sizes(num_groups: int, total: int, gen: torch.Generator) -> torch.Tensor:
    # Uneven sizes with ~10% empty groups (empty experts happen in real routing).
    probs = torch.rand(num_groups, generator=gen)
    probs[torch.rand(num_groups, generator=gen) < 0.1] = 0
    sizes = torch.multinomial(probs / probs.sum(), total, replacement=True, generator=gen).bincount(
        minlength=num_groups
    )
    assert sizes.sum().item() == total
    return sizes.to(torch.int64)


def check_raw(num_groups: int, batch_sizes_host: torch.Tensor, gen: torch.Generator) -> None:
    d, h = 2048, 256
    total = int(batch_sizes_host.sum())
    x0 = torch.randn(total, d, generator=gen).to("cuda", torch.bfloat16)
    for trans_b in (True, False):
        w_shape = (num_groups, h, d) if trans_b else (num_groups, d, h)
        w0 = (torch.randn(*w_shape, generator=gen) * 0.02).to("cuda", torch.bfloat16)
        x_ref, w_ref = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        y_ref = loop_gmm(x_ref, w_ref, batch_sizes_host, trans_b)
        g = torch.randn(y_ref.shape, generator=gen).to("cuda", torch.bfloat16)
        y_ref.backward(g)

        x_h, w_h = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        y_h = grouped_gemm.ops.gmm(x_h, w_h, batch_sizes_host, trans_b=trans_b)
        y_h.backward(g)
        tag = f"raw gmm {num_groups}g host bs trans_b={trans_b}"
        close(f"{tag} forward", y_h, y_ref, 2e-2, 1e-1)
        close(f"{tag} grad x", x_h.grad, x_ref.grad, 2e-2, 1e-1)
        close(f"{tag} grad w", w_h.grad, w_ref.grad, 2e-2, 1e-1)

        if num_groups <= 512:
            x_d, w_d = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
            y_d = grouped_gemm.ops.gmm(x_d, w_d, batch_sizes_host.cuda(), trans_b=trans_b)
            y_d.backward(g)
            tag = f"raw gmm {num_groups}g host-vs-device bs trans_b={trans_b}"
            close(f"{tag} forward", y_d, y_h, 0.0, 0.0)
            close(f"{tag} grad x", x_d.grad, x_h.grad, 0.0, 0.0)
            close(f"{tag} grad w", w_d.grad, w_h.grad, 0.0, 0.0)
        else:
            try:
                grouped_gemm.ops.gmm(x0, w0, batch_sizes_host.cuda(), trans_b=trans_b)
                print(f"  ??? {num_groups} groups with DEVICE bs did NOT raise (limit gone?)")
            except RuntimeError as e:
                print(
                    f"  OK  {num_groups} groups with DEVICE bs raises as expected: {str(e)[:60]}..."
                )


def check_module(num_experts: int, batch_sizes_host: torch.Tensor, gen: torch.Generator) -> None:
    d, h = 2048, 256
    torch.manual_seed(0)
    mlp = DroplessMoEMLP(d_model=d, hidden_size=h, num_experts=num_experts, init_device="cuda")
    assert mlp._gmm is not None, "grouped_gemm not picked up by DroplessMoEMLP"
    mlp = mlp.to(torch.bfloat16)
    total = int(batch_sizes_host.sum())
    x0 = torch.randn(total, d, generator=gen).to("cuda", torch.bfloat16)
    bs_dev = batch_sizes_host.cuda()  # what the router hands the MLP in training
    g = torch.randn(total, d, generator=gen).to("cuda", torch.bfloat16)

    x_g = x0.clone().requires_grad_(True)
    y_g = mlp(x_g, bs_dev)
    assert bs_dev.is_cuda, "module must not mutate the caller's tensor"
    y_g.backward(g)
    grads_g = [p.grad.clone() for p in (mlp.w1, mlp.w2, mlp.w3)]
    mlp.zero_grad(set_to_none=True)

    real_gmm, mlp._gmm = mlp._gmm, None  # module's own python-loop fallback as reference
    x_l = x0.clone().requires_grad_(True)
    y_l = mlp(x_l, bs_dev)
    y_l.backward(g)
    grads_l = [p.grad.clone() for p in (mlp.w1, mlp.w2, mlp.w3)]
    mlp._gmm = real_gmm

    tag = f"DroplessMoEMLP {num_experts}e grouped_gemm-vs-loop"
    close(f"{tag} forward", y_g, y_l, 2e-2, 1e-1)
    close(f"{tag} grad x", x_g.grad, x_l.grad, 2e-2, 1e-1)
    for name, a, b in zip(("w1", "w2", "w3"), grads_g, grads_l):
        close(f"{tag} grad {name}", a, b, 2e-2, 1e-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_known_args()  # ignore run_name / --data-root from launch_common
    print(
        "torch",
        torch.__version__,
        "| grouped_gemm",
        getattr(grouped_gemm, "__version__", "?"),
        "|",
        torch.cuda.get_device_name(0),
    )
    gen = torch.Generator().manual_seed(1234)
    for num_groups in (512, 1024):
        # ~ production rank micro-batch: 4 x 4096 tokens x top_k 8 assignments
        bs = random_batch_sizes(num_groups, 4 * 4096 * 8, gen)
        print(
            f"== {num_groups} groups: {int((bs == 0).sum())} empty, max {int(bs.max())} tokens/group"
        )
        check_raw(num_groups, bs, gen)
    bs = random_batch_sizes(1024, 4 * 4096 * 8, gen)
    print("== DroplessMoEMLP end-to-end, 1024 experts")
    check_module(1024, bs, gen)
    if FAILS:
        print(f"\nFAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
