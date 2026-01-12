# fused_adafactor_2pass.py
import math
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.optim import Optimizer


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# -----------------------------------------------------------------------------
# Pass 1: accumulate sum(g^2) by row and by col + total sum(g^2)
# -----------------------------------------------------------------------------
@triton.jit
def _adafactor_stats_kernel(
    g_ptr,
    row_sum_ptr,
    col_sum_ptr,
    g2_sum_ptr,  # scalar
    M: tl.constexpr,
    N: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]

    g = tl.load(
        g_ptr + rm[:, None] * stride_gm + rn[None, :] * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    g2 = g * g

    # partial sums for this tile
    row_part = tl.sum(g2, axis=1)  # [BM]
    col_part = tl.sum(g2, axis=0)  # [BN]
    tile_sum = tl.sum(g2, axis=None)  # scalar

    # atomically accumulate into global row/col sums
    tl.atomic_add(row_sum_ptr + rm, row_part, mask=mask_m)
    tl.atomic_add(col_sum_ptr + rn, col_part, mask=mask_n)
    tl.atomic_add(g2_sum_ptr, tile_sum)


# -----------------------------------------------------------------------------
# Pass 2: update weights using completed row/col EMAs and clip scale
# -----------------------------------------------------------------------------
@triton.jit
def _adafactor_update_kernel(
    p_ptr,
    g_ptr,
    r_ptr,
    c_ptr,
    r_mean_inv_ptr,  # scalar
    scale_ptr,  # scalar
    lr: tl.constexpr,
    eps2: tl.constexpr,
    weight_decay: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]

    r = tl.load(r_ptr + rm, mask=mask_m, other=0.0).to(tl.float32)  # [BM]
    c = tl.load(c_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)  # [BN]

    r_mean_inv = tl.load(r_mean_inv_ptr).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    # v_ij = r_i * c_j / mean(r) (mean(r) inverted)
    v = (r[:, None] * c[None, :]) * r_mean_inv

    g = tl.load(
        g_ptr + rm[:, None] * stride_gm + rn[None, :] * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g = g * scale

    # weight update (compute in fp32)
    p = tl.load(
        p_ptr + rm[:, None] * stride_pm + rn[None, :] * stride_pn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    u = g * tl.rsqrt(v + eps2)
    p_new = p * (1.0 - lr * weight_decay) - lr * u

    # store back (bf16/fp16 supported; fp32 also OK)
    tl.store(
        p_ptr + rm[:, None] * stride_pm + rn[None, :] * stride_pn,
        p_new.to(tl.bfloat16),
        mask=mask,
    )


class FusedAdafactor2Pass(Optimizer):
    """
    2-pass Triton Adafactor for 2D tensors:
      Pass 1: atomic row/col g^2 sums + global g^2 sum (for clipping)
      Update EMA row/col on GPU with torch ops (no CPU sync)
      Pass 2: weight update with reconstructed v

    Notes:
      - Keeps row/col states in FP32 (good).
      - Updates params in-place in BF16 (compute in FP32).
      - Reduces grad reads from 3 -> 2 for 2D tensors.
      - Leaves 1D tensors on a simpler path (can keep your existing 1D kernel).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
        block_m: int = 32,
        block_n: int = 128,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            weight_decay=weight_decay,
            block_m=block_m,
            block_n=block_n,
        )
        super().__init__(params, defaults)

    def _get_rho(self, step: int, decay_rate: float) -> float:
        return min(0.999, 1.0 - math.pow(step + 1, decay_rate))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            _, eps2 = group["eps"]
            clip_threshold = float(group["clip_threshold"])
            decay_rate = float(group["decay_rate"])
            weight_decay = float(group["weight_decay"])
            BLOCK_M = int(group["block_m"])
            BLOCK_N = int(group["block_n"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse grads")

                grad = p.grad
                if grad.device.type != "cuda":
                    raise RuntimeError("This fused optimizer is CUDA-only")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1
                rho = self._get_rho(state["step"], decay_rate)

                if grad.ndim < 2:
                    # Keep your existing 1D kernel or just do a simple fallback.
                    # (Not included here; 1D is not the performance-critical path.)
                    g = grad.float()
                    rms = torch.linalg.vector_norm(g) / math.sqrt(g.numel())
                    scale = (clip_threshold / (rms + 1e-12)).clamp(max=1.0)
                    g = g * scale
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                    v = state["exp_avg_sq"]
                    v.mul_(rho).addcmul_(g, g, value=1.0 - rho)
                    u = g * torch.rsqrt(v + eps2)
                    p.mul_(1.0 - lr * weight_decay).add_(u.to(p.dtype), alpha=-lr)
                    continue

                # Flatten to (M, N) as your original code did
                orig_shape = p.shape
                N = orig_shape[-1]
                M = int(p.numel() // N)

                p2 = p.view(M, N)
                g2 = grad.view(M, N)

                if not p2.is_contiguous():
                    raise RuntimeError(
                        "Param not contiguous after view; add a contiguous layout "
                        "or handle strides in the kernel."
                    )
                if not g2.is_contiguous():
                    g2 = g2.contiguous()

                # State buffers (FP32) for Adafactor
                if "exp_avg_sq_row" not in state:
                    state["exp_avg_sq_row"] = torch.zeros(
                        (M,), device=p.device, dtype=torch.float32
                    )
                    state["exp_avg_sq_col"] = torch.zeros(
                        (N,), device=p.device, dtype=torch.float32
                    )

                    # Temp buffers for this param (reuse each step)
                    state["row_sum_buf"] = torch.zeros(
                        (M,), device=p.device, dtype=torch.float32
                    )
                    state["col_sum_buf"] = torch.zeros(
                        (N,), device=p.device, dtype=torch.float32
                    )
                    state["g2_sum_buf"] = torch.zeros(
                        (), device=p.device, dtype=torch.float32
                    )
                    state["scale_buf"] = torch.ones(
                        (), device=p.device, dtype=torch.float32
                    )
                    state["r_mean_inv_buf"] = torch.ones(
                        (), device=p.device, dtype=torch.float32
                    )

                r = state["exp_avg_sq_row"]
                c = state["exp_avg_sq_col"]
                row_sum = state["row_sum_buf"]
                col_sum = state["col_sum_buf"]
                g2_sum = state["g2_sum_buf"]
                scale_buf = state["scale_buf"]
                r_mean_inv_buf = state["r_mean_inv_buf"]

                row_sum.zero_()
                col_sum.zero_()
                g2_sum.zero_()

                grid = (_ceil_div(M, BLOCK_M), _ceil_div(N, BLOCK_N))
                _adafactor_stats_kernel[grid](
                    g2,
                    row_sum,
                    col_sum,
                    g2_sum,
                    M=M,
                    N=N,
                    stride_gm=g2.stride(0),
                    stride_gn=g2.stride(1),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                )

                # All on-GPU: compute means, update EMAs
                row_mean = row_sum / float(N)
                col_mean = col_sum / float(M)
                r.mul_(rho).add_(row_mean, alpha=1.0 - rho)
                c.mul_(rho).add_(col_mean, alpha=1.0 - rho)

                # grad RMS for clipping (global per-tensor)
                numel = float(M * N)
                grad_rms = torch.sqrt(g2_sum / numel)
                scale = (clip_threshold / (grad_rms + 1e-12)).clamp(max=1.0)
                scale_buf.copy_(scale)

                # mean(r) inverse (device scalar) for v reconstruction
                r_mean_inv = 1.0 / (r.mean() + 1e-30)
                r_mean_inv_buf.copy_(r_mean_inv)

                _adafactor_update_kernel[grid](
                    p2,
                    g2,
                    r,
                    c,
                    r_mean_inv_buf,
                    scale_buf,
                    lr=lr,
                    eps2=eps2,
                    weight_decay=weight_decay,
                    M=M,
                    N=N,
                    stride_pm=p2.stride(0),
                    stride_pn=p2.stride(1),
                    stride_gm=g2.stride(0),
                    stride_gn=g2.stride(1),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                )

        return loss