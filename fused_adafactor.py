# fused_adafactor.py
import math
import torch
import triton
import triton.language as tl
from torch.optim import Optimizer


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def _adafactor_1d_kernel(
    p_ptr,
    g_ptr,
    v_ptr,
    lr,
    rho,
    eps2,
    weight_decay,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Adafactor for 1D params (biases, RMSNorm weights)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    p = tl.load(p_ptr + offs, mask=mask)
    g = tl.load(g_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)

    g2 = g * g
    v_new = rho * v + (1.0 - rho) * g2
    u = g * tl.rsqrt(v_new + eps2)
    p_new = p * (1.0 - lr * weight_decay) - lr * u

    tl.store(v_ptr + offs, v_new, mask=mask)
    tl.store(p_ptr + offs, p_new, mask=mask)


@triton.jit
def _update_row_kernel(
    g_ptr,
    r_ptr,
    rho,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """Update row factor r[m] = ρ*r[m] + (1-ρ)*mean(g[m,:]²)"""
    row = tl.program_id(0)
    if row >= M:
        return

    # Accumulate sum of g² across this row
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        g = tl.load(g_ptr + row * N + offs, mask=mask, other=0.0)
        acc += tl.sum(g * g)

    mean_g2 = acc / N
    r_old = tl.load(r_ptr + row)
    r_new = rho * r_old + (1.0 - rho) * mean_g2
    tl.store(r_ptr + row, r_new)


@triton.jit
def _update_col_kernel(
    g_ptr,
    c_ptr,
    rho,
    M,
    N,
    BLOCK_M: tl.constexpr,
):
    """Update col factor c[n] = ρ*c[n] + (1-ρ)*mean(g[:,n]²)"""
    col = tl.program_id(0)
    if col >= N:
        return

    acc = 0.0
    for start in range(0, M, BLOCK_M):
        offs = start + tl.arange(0, BLOCK_M)
        mask = offs < M
        g = tl.load(g_ptr + offs * N + col, mask=mask, other=0.0)
        acc += tl.sum(g * g)

    mean_g2 = acc / M
    c_old = tl.load(c_ptr + col)
    c_new = rho * c_old + (1.0 - rho) * mean_g2
    tl.store(c_ptr + col, c_new)


@triton.jit
def _adafactor_2d_update_kernel(
    p_ptr,
    g_ptr,
    r_ptr,
    c_ptr,
    lr,
    eps2,
    weight_decay,
    r_mean_inv_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused weight update for 2D params. Reconstructs v on-the-fly."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    
    r_mean_inv = tl.load(r_mean_inv_ptr)

    r = tl.load(r_ptr + offs_m, mask=mask_m, other=0.0)
    c = tl.load(c_ptr + offs_n, mask=mask_n, other=0.0)

    # v_ij = r_i * c_j / mean(r)
    v = (r[:, None] * c[None, :]) * r_mean_inv

    offs_2d = offs_m[:, None] * N + offs_n[None, :]
    mask_2d = mask_m[:, None] & mask_n[None, :]

    p = tl.load(p_ptr + offs_2d, mask=mask_2d, other=0.0)
    g = tl.load(g_ptr + offs_2d, mask=mask_2d, other=0.0)

    u = g * tl.rsqrt(v + eps2)
    p_new = p * (1.0 - lr * weight_decay) - lr * u

    tl.store(p_ptr + offs_2d, p_new, mask=mask_2d)


# =============================================================================
# Fused Adafactor Optimizer
# =============================================================================

class FusedAdafactor(Optimizer):
    """
    Fused Adafactor using Triton kernels.
    
    Matches HuggingFace Adafactor with:
      - beta1=None (no momentum)
      - relative_step=False
      - scale_parameter=False
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _get_rho(self, step: int, decay_rate: float) -> float:
        return min(0.999, 1.0 - math.pow(step + 1, decay_rate))

    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        # returns 0-dim tensor on device (no CPU sync)
        return torch.linalg.vector_norm(tensor) / math.sqrt(tensor.numel())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps1, eps2 = group["eps"]
            clip_threshold = group["clip_threshold"]
            decay_rate = group["decay_rate"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedAdafactor does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    shape = grad.shape
                    if len(shape) >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            shape[:-1], dtype=torch.float32, device=grad.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            shape[:-2] + shape[-1:], dtype=torch.float32, device=grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                rho = self._get_rho(step, decay_rate)

                grad_fp32 = grad.float().contiguous()

                if len(grad.shape) >= 2:
                    self._step_2d(p, grad_fp32, state, lr, rho, eps2, weight_decay, clip_threshold)
                else:
                    self._step_1d(p, grad_fp32, state, lr, rho, eps2, weight_decay, clip_threshold)

        return loss

    def _step_1d(self, p, grad, state, lr, rho, eps2, weight_decay, clip_threshold):
        v = state["exp_avg_sq"]
        n = p.numel()

        grad_rms = self._rms(grad)
        scale = (clip_threshold / (grad_rms + 1e-12)).clamp(max=1.0)
        grad = grad * scale

        p_data = p.data.float().contiguous()

        BLOCK_SIZE = 1024
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _adafactor_1d_kernel[grid](
            p_data, grad, v,
            lr, rho, eps2, weight_decay, n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        p.data.copy_(p_data)

    def _step_2d(self, p, grad, state, lr, rho, eps2, weight_decay, clip_threshold):
        r = state["exp_avg_sq_row"]
        c = state["exp_avg_sq_col"]

        orig_shape = p.shape
        p_fp32 = p.data.float().contiguous()
        p_2d = p_fp32.view(-1, orig_shape[-1])
        g_2d = grad.view(-1, orig_shape[-1]).contiguous()

        M, N = p_2d.shape
        r_flat = r.view(-1).contiguous()
        c_flat = c.view(-1).contiguous()

        # --- Update row factors ---
        BLOCK_N = min(1024, triton.next_power_of_2(N))
        _update_row_kernel[(M,)](
            g_2d, r_flat, rho, M, N,
            BLOCK_N=BLOCK_N,
        )

        # --- Update col factors ---
        BLOCK_M = min(1024, triton.next_power_of_2(M))
        _update_col_kernel[(N,)](
            g_2d, c_flat, rho, M, N,
            BLOCK_M=BLOCK_M,
        )

        # no CPU sync: keep these as device scalars
        r_mean_inv = 1.0 / (r_flat.mean(dtype=torch.float32) + 1e-30)
        grad_rms = self._rms(g_2d)
        scale = (clip_threshold / (grad_rms + 1e-12)).clamp(max=1.0)
        g_2d = g_2d * scale

        # --- Fused weight update ---
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (
            (M + BLOCK_M - 1) // BLOCK_M,
            (N + BLOCK_N - 1) // BLOCK_N,
        )

        _adafactor_2d_update_kernel[grid](
            p_2d, g_2d, r_flat, c_flat,
            lr, eps2, weight_decay, r_mean_inv,
            M, N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        p.data.copy_(p_fp32.view(orig_shape))


def get_fused_adafactor(params, **kwargs) -> FusedAdafactor:
    return FusedAdafactor(params, **kwargs)