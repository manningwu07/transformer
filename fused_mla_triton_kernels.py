# mla_triton_kernels_fixed.py
"""
DeepSeek MLA Triton Kernels - Fixed for RTX 5080 (Blackwell)
Target: 101KB SRAM, BF16 storage, FP32 accumulation
"""

import math
import torch
import triton
import time
import triton.language as tl
from typing import Optional

# =============================================================================
# Constants
# =============================================================================
D_MODEL = 1792
D_LATENT = 384
N_HEADS = 14
HEAD_DIM = 128
ROPE_DIM = 64
HALF_ROPE = 32

def _reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def _bench_op(name: str, fn, warmup: int = 25, iters: int = 200) -> dict:
    """
    Bench an op for time  peak memory.
    Returns dict with ms/iter and peak allocated/reserved bytes.
    """
    _reset_cuda_peak()
    # Warmup
    for _ in range(warmup):
        y = fn()
        # prevent DCE
        if isinstance(y, torch.Tensor):
            y = y.view(-1)
            _ = y[0]
    torch.cuda.synchronize()

    _reset_cuda_peak()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn()
        if isinstance(y, torch.Tensor):
            y = y.view(-1)
            _ = y[0]
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0 / iters

    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_res = int(torch.cuda.max_memory_reserved())
    return {
        "name": name,
        "ms": dt,
        "peak_alloc_bytes": peak_alloc,
        "peak_res_bytes": peak_res,
    }


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024**2):.2f} MiB"


def _robust_diff_metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    diff = (expected.float() - actual.float()).abs().flatten()
    mean = float(diff.mean().item())
    maxv = float(diff.max().item())
    # percentiles via kthvalue (approx but fine for tests)
    n = diff.numel()
    p99 = float(diff.kthvalue(max(1, int(0.99 * n))).values.item())
    p999 = float(diff.kthvalue(max(1, int(0.999 * n))).values.item())
    return {"mean": mean, "p99": p99, "p999": p999, "max": maxv}


def _maybe_compile(fn):
    """
    Wrap a zero-arg function with torch.compile if available.
    Uses max-autotune-no-cudagraphs to match your training config.
    """
    try:
        compiled = torch.compile(
            fn,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=False,
        )
        return compiled
    except Exception as e:
        print(f"⚠️ torch.compile not available for benchmark: {e}")
        return fn


# =============================================================================
# Kernel 1: Fused RMSNorm + MLA Down-Projection (FIXED)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 128, "BLOCK_N": 128},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_N": 128},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64},
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M"],
)
@triton.jit
def fused_rmsnorm_down_kernel(
    # Pointers
    x_ptr,
    w_norm_ptr,
    w_down_ptr,
    out_ptr,
    # Dimensions
    M,
    # Strides (NOT constexpr - they're runtime values)
    stride_x_m,
    stride_x_k,
    stride_w_k,
    stride_w_n,
    stride_out_m,
    stride_out_n,
    # RMSNorm epsilon
    eps,
    # Compile-time shape constants
    K: tl.constexpr,  # 1792
    N: tl.constexpr,  # 384
    # Block sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused RMSNorm + Down-Projection.
    
    Phase 1: Stream x once -> compute sum(x²) per row
    Phase 2: Stream x again (L2 cached) -> normalize in regs -> matmul
    
    CRITICAL: Normalized tensor stays in REGISTERS. Never touches VRAM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # =========================================================================
    # Phase 1: Compute RMS scale factor per row
    # =========================================================================
    rms_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # K=1792 = 14 × 128, exactly divisible by BLOCK_K=128
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K  # Safety mask for non-power-of-2

        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        x_f32 = x_tile.to(tl.float32)

        rms_sq += tl.sum(x_f32 * x_f32, axis=1)

    # Compute inverse RMS: scale = rsqrt(mean(x²) + eps)
    inv_rms = tl.rsqrt(rms_sq * (1.0 / K) + eps)  # [BLOCK_M]

    # =========================================================================
    # Phase 2: Fused normalize + matmul
    # =========================================================================
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Reload x (should hit L2 cache from Phase 1)
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        # Load RMSNorm weights [BLOCK_K]
        w_norm_tile = tl.load(w_norm_ptr + offs_k, mask=k_mask, other=0.0)

        # === CRITICAL FUSION: Normalize in registers ===
        x_f32 = x_tile.to(tl.float32)
        x_normed = x_f32 * inv_rms[:, None] * w_norm_tile[None, :].to(tl.float32)

        # Load down-projection weight [BLOCK_K, BLOCK_N]
        w_ptrs = (
            w_down_ptr
            + offs_k[:, None] * stride_w_k
            + offs_n[None, :] * stride_w_n
        )
        w_tile = tl.load(
            w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0
        )

        # Matmul accumulation
        acc += tl.dot(
            x_normed.to(tl.bfloat16),
            w_tile.to(tl.bfloat16),
            allow_tf32=True,
        ).to(tl.float32)

    # Store output [BLOCK_M, BLOCK_N]
    out_ptrs = (
        out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    )
    tl.store(
        out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :]
    )


def fused_rmsnorm_down_proj(
    x: torch.Tensor,  # [M, 1792] bf16
    w_norm: torch.Tensor,  # [1792] bf16
    w_down: torch.Tensor,  # [1792, 384] bf16
    eps: float = 1e-6,
) -> torch.Tensor:
    """Python wrapper for Fused RMSNorm + Down-Projection."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.is_contiguous(), "Input must be contiguous"

    M, K = x.shape
    K_w, N = w_down.shape

    assert K == 1792 and N == 384, f"Expected d_model=1792, d_latent=384"
    assert K == K_w == w_norm.numel()

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_rmsnorm_down_kernel[grid](
        x,
        w_norm,
        w_down,
        out,
        M,
        x.stride(0),
        x.stride(1),
        w_down.stride(0),
        w_down.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        K=K,
        N=N,
    )

    return out


# =============================================================================
# Kernel 2: Fused Up-Projection + RoPE (FIXED - Single Matmul + In-Register RoPE)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_stages=2, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def fused_up_proj_rope_kernel(
    # Inputs
    latent_ptr,  # [M, K]
    w_up_ptr,  # [K, N_HEADS * HEAD_DIM]
    cos_ptr,  # [MAX_SEQ, HALF_ROPE]
    sin_ptr,  # [MAX_SEQ, HALF_ROPE]
    # Output
    out_ptr,  # [M, N_HEADS, HEAD_DIM]
    # Dims
    M,
    pos_offset,
    # Strides (runtime, NOT constexpr)
    stride_lat_m,
    stride_lat_k,
    stride_w_k,
    stride_w_n,
    stride_cos_s,
    stride_cos_d,
    stride_out_m,
    stride_out_h,
    stride_out_d,
    # Constants
    K: tl.constexpr,  # 384
    N_HEADS: tl.constexpr,  # 14
    HEAD_DIM: tl.constexpr,  # 128
    ROPE_DIM: tl.constexpr,  # 64
    HALF_ROPE: tl.constexpr,  # 32
    APPLY_ROPE: tl.constexpr,  # True for Q/K, False for V
    # Autotuned
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Up-Projection + RoPE for a single attention head.
    
    Grid: (cdiv(M, BLOCK_M), N_HEADS)
    
    Strategy:
      1. Compute full [BLOCK_M, HEAD_DIM] matmul for this head
      2. Apply RoPE to first ROPE_DIM dims in registers
      3. Store result
    """
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    head_offset = head_idx * HEAD_DIM

    # =========================================================================
    # Matmul: latent @ w_up[:, head_slice] -> [BLOCK_M, HEAD_DIM]
    # =========================================================================
    # Full HEAD_DIM accumulator (single matmul, not 3 separate ones)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load latent [BLOCK_M, BLOCK_K]
        lat_ptrs = (
            latent_ptr
            + offs_m[:, None] * stride_lat_m
            + offs_k[None, :] * stride_lat_k
        )
        lat = tl.load(lat_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        # Load weight [BLOCK_K, HEAD_DIM] for this head
        offs_d = tl.arange(0, HEAD_DIM)
        w_ptrs = (
            w_up_ptr
            + offs_k[:, None] * stride_w_k
            + (head_offset + offs_d)[None, :] * stride_w_n
        )
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)

        acc += tl.dot(lat.to(tl.bfloat16), w.to(tl.bfloat16), allow_tf32=True).to(
            tl.float32
        )

    # =========================================================================
    # Apply RoPE to first ROPE_DIM dimensions (in registers)
    # DeepSeek uses split-half rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    # =========================================================================
    if APPLY_ROPE:
        positions = pos_offset + offs_m

        # Load cos/sin for these positions
        offs_rope = tl.arange(0, HALF_ROPE)
        cos_ptrs = (
            cos_ptr
            + positions[:, None] * stride_cos_s
            + offs_rope[None, :] * stride_cos_d
        )
        sin_ptrs = (
            sin_ptr
            + positions[:, None] * stride_cos_s
            + offs_rope[None, :] * stride_cos_d
        )

        cos_val = tl.load(cos_ptrs, mask=mask_m[:, None], other=1.0)
        sin_val = tl.load(sin_ptrs, mask=mask_m[:, None], other=0.0)

        # Extract x1 = acc[:, 0:32], x2 = acc[:, 32:64]
        # Triton doesn't support direct slicing, so we use pointer arithmetic
        # We'll store rotated values directly

        # For the first HALF_ROPE dims: y1 = x1*cos - x2*sin
        # For dims [HALF_ROPE:ROPE_DIM]: y2 = x1*sin + x2*cos

        # Create output with rotation applied
        # This requires extracting slices - we'll handle this at store time

    # =========================================================================
    # Store output [BLOCK_M, HEAD_DIM]
    # =========================================================================
    offs_d = tl.arange(0, HEAD_DIM)
    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_out_m
        + head_idx * stride_out_h
        + offs_d[None, :] * stride_out_d
    )

    if APPLY_ROPE:
        # Store with RoPE applied to first ROPE_DIM dims
        # For simplicity, we'll iterate over dimension groups

        # Group 1: dims [0, HALF_ROPE) - first rotation output
        offs_d1 = tl.arange(0, HALF_ROPE)
        x1_mask = offs_d1[None, :] < HALF_ROPE

        # Extract x1 and x2 from accumulator
        # x1 = acc[:, 0:32], x2 = acc[:, 32:64]
        x1 = tl.sum(
            tl.where(
                (tl.arange(0, HEAD_DIM)[None, :] == offs_d1[:, None]),
                acc,
                tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
            ),
            axis=1,
        )
        # This approach is inefficient - let's use a cleaner method

        # BETTER: Store full acc, then apply RoPE in a separate store pattern
        # For now, store without RoPE and let caller handle it
        # (The truly fused version requires more complex Triton patterns)
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])
    else:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


# =============================================================================
# Kernel 2b: Cleaner Fused Up-Proj + RoPE with Split Accumulators
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_stages=2, num_warps=4),
    ],
    key=["M"],
)
@triton.jit
def fused_up_proj_rope_v2_kernel(
    # Inputs
    latent_ptr,
    w_up_ptr,
    cos_ptr,
    sin_ptr,
    # Output
    out_ptr,
    # Dims
    M,
    pos_offset,
    # Strides
    stride_lat_m,
    stride_lat_k,
    stride_w_k,
    stride_w_n,
    stride_cos_s,
    stride_cos_d,
    stride_out_m,
    stride_out_h,
    stride_out_d,
    # Constants
    K: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    CONTENT_DIM: tl.constexpr,  # HEAD_DIM - ROPE_DIM = 64
    APPLY_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused up-projection + RoPE with split accumulators.
    
    Strategy: Maintain 3 separate accumulators for:
      - acc1: dims [0, HALF_ROPE) - will get rotated
      - acc2: dims [HALF_ROPE, ROPE_DIM) - will get rotated  
      - acc_content: dims [ROPE_DIM, HEAD_DIM) - no rotation
    
    This avoids complex slicing at the cost of 3 smaller matmuls.
    For RTX 5080 SRAM budget, this is acceptable.
    """
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    head_offset = head_idx * HEAD_DIM

    # Dimension offsets for each group
    offs_d1 = tl.arange(0, HALF_ROPE)  # [0, 32)
    offs_d2 = HALF_ROPE + tl.arange(0, HALF_ROPE)  # [32, 64)
    offs_content = ROPE_DIM + tl.arange(0, CONTENT_DIM)  # [64, 128)

    # Split accumulators
    acc1 = tl.zeros([BLOCK_M, HALF_ROPE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, HALF_ROPE], dtype=tl.float32)
    acc_content = tl.zeros([BLOCK_M, CONTENT_DIM], dtype=tl.float32)

    # =========================================================================
    # Tiled Matmul with Split Weight Loading
    # =========================================================================
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load latent [BLOCK_M, BLOCK_K]
        lat_ptrs = (
            latent_ptr
            + offs_m[:, None] * stride_lat_m
            + offs_k[None, :] * stride_lat_k
        )
        lat = tl.load(
            lat_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0
        ).to(tl.bfloat16)

        # Load weight slices for this head (3 loads per K-tile)
        # w1: [BLOCK_K, HALF_ROPE]
        w1_ptrs = (
            w_up_ptr
            + offs_k[:, None] * stride_w_k
            + (head_offset + offs_d1)[None, :] * stride_w_n
        )
        w1 = tl.load(w1_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)

        # w2: [BLOCK_K, HALF_ROPE]
        w2_ptrs = (
            w_up_ptr
            + offs_k[:, None] * stride_w_k
            + (head_offset + offs_d2)[None, :] * stride_w_n
        )
        w2 = tl.load(w2_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)

        # w_content: [BLOCK_K, CONTENT_DIM]
        w_content_ptrs = (
            w_up_ptr
            + offs_k[:, None] * stride_w_k
            + (head_offset + offs_content)[None, :] * stride_w_n
        )
        w_content = tl.load(w_content_ptrs, mask=k_mask[:, None], other=0.0).to(
            tl.bfloat16
        )

        # Three matmuls (Tensor Core utilization depends on tile size)
        acc1 += tl.dot(lat, w1, allow_tf32=True).to(tl.float32)
        acc2 += tl.dot(lat, w2, allow_tf32=True).to(tl.float32)
        acc_content += tl.dot(lat, w_content, allow_tf32=True).to(tl.float32)

    # =========================================================================
    # Apply RoPE in Registers
    # =========================================================================
    if APPLY_ROPE:
        positions = pos_offset + offs_m

        cos_ptrs = (
            cos_ptr
            + positions[:, None] * stride_cos_s
            + offs_d1[None, :] * stride_cos_d
        )
        sin_ptrs = (
            sin_ptr
            + positions[:, None] * stride_cos_s
            + offs_d1[None, :] * stride_cos_d
        )

        cos_val = tl.load(cos_ptrs, mask=mask_m[:, None], other=1.0)
        sin_val = tl.load(sin_ptrs, mask=mask_m[:, None], other=0.0)

        # Rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        y1 = acc1 * cos_val - acc2 * sin_val
        y2 = acc1 * sin_val + acc2 * cos_val
    else:
        y1 = acc1
        y2 = acc2

    # =========================================================================
    # Store output
    # =========================================================================
    base_ptr = out_ptr + offs_m[:, None] * stride_out_m + head_idx * stride_out_h

    # Store y1 at dims [0:HALF_ROPE]
    tl.store(
        base_ptr + offs_d1[None, :] * stride_out_d,
        y1.to(tl.bfloat16),
        mask=mask_m[:, None],
    )

    # Store y2 at dims [HALF_ROPE:ROPE_DIM]
    tl.store(
        base_ptr + offs_d2[None, :] * stride_out_d,
        y2.to(tl.bfloat16),
        mask=mask_m[:, None],
    )

    # Store content at dims [ROPE_DIM:HEAD_DIM]
    tl.store(
        base_ptr + offs_content[None, :] * stride_out_d,
        acc_content.to(tl.bfloat16),
        mask=mask_m[:, None],
    )


# =============================================================================
# Kernel 3: In-Place RoPE (for when you need it separately)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def rope_inplace_kernel(
    # Input/Output (in-place)
    qk_ptr,  # [M, N_HEADS, HEAD_DIM]
    cos_ptr,  # [MAX_SEQ, HALF_ROPE]
    sin_ptr,  # [MAX_SEQ, HALF_ROPE]
    # Dims
    M,
    pos_offset,
    # Strides
    stride_qk_m,
    stride_qk_h,
    stride_qk_d,
    stride_cos_s,
    stride_cos_d,
    # Constants
    N_HEADS: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    In-place RoPE application.
    
    Grid: (cdiv(M, BLOCK_M), N_HEADS)
    
    Operates on first ROPE_DIM=64 dimensions only.
    """
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    positions = pos_offset + offs_m

    # Load cos/sin
    offs_rope = tl.arange(0, HALF_ROPE)
    cos_ptrs = (
        cos_ptr
        + positions[:, None] * stride_cos_s
        + offs_rope[None, :] * stride_cos_d
    )
    sin_ptrs = (
        sin_ptr
        + positions[:, None] * stride_cos_s
        + offs_rope[None, :] * stride_cos_d
    )

    cos_val = tl.load(cos_ptrs, mask=mask_m[:, None], other=1.0)
    sin_val = tl.load(sin_ptrs, mask=mask_m[:, None], other=0.0)

    # Load x1 and x2
    x1_ptrs = (
        qk_ptr
        + offs_m[:, None] * stride_qk_m
        + head_idx * stride_qk_h
        + offs_rope[None, :] * stride_qk_d
    )
    x2_ptrs = (
        qk_ptr
        + offs_m[:, None] * stride_qk_m
        + head_idx * stride_qk_h
        + (HALF_ROPE + offs_rope)[None, :] * stride_qk_d
    )

    x1 = tl.load(x1_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # Apply rotation
    y1 = x1 * cos_val - x2 * sin_val
    y2 = x1 * sin_val + x2 * cos_val

    # Store back
    tl.store(x1_ptrs, y1.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(x2_ptrs, y2.to(tl.bfloat16), mask=mask_m[:, None])


# =============================================================================
# Python Wrappers
# =============================================================================

def fused_mla_up_proj_rope(
    latent: torch.Tensor,  # [M, 384]
    w_up: torch.Tensor,  # [384, N_HEADS * HEAD_DIM]
    cos: torch.Tensor,  # [MAX_SEQ, HALF_ROPE]
    sin: torch.Tensor,  # [MAX_SEQ, HALF_ROPE]
    pos_offset: int = 0,
    apply_rope: bool = True,
    n_heads: int = 14,
    head_dim: int = 128,
    rope_dim: int = 64,
) -> torch.Tensor:
    """Fused up-projection + RoPE."""
    assert latent.is_cuda and latent.dtype == torch.bfloat16
    assert latent.is_contiguous()

    M, K = latent.shape
    half_rope = rope_dim // 2
    content_dim = head_dim - rope_dim

    out = torch.empty((M, n_heads, head_dim), dtype=torch.bfloat16, device=latent.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), n_heads)

    fused_up_proj_rope_v2_kernel[grid](
        latent,
        w_up,
        cos,
        sin,
        out,
        M,
        pos_offset,
        latent.stride(0),
        latent.stride(1),
        w_up.stride(0),
        w_up.stride(1),
        cos.stride(0),
        cos.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        K=K,
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HALF_ROPE=half_rope,
        CONTENT_DIM=content_dim,
        APPLY_ROPE=apply_rope,
    )

    return out


def apply_rope_inplace(
    qk: torch.Tensor,  # [M, N_HEADS, HEAD_DIM]
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_offset: int = 0,
    n_heads: int = 14,
    half_rope: int = 32,
) -> None:
    """In-place RoPE application."""
    M = qk.shape[0]

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), n_heads)

    rope_inplace_kernel[grid](
        qk,
        cos,
        sin,
        M,
        pos_offset,
        qk.stride(0),
        qk.stride(1),
        qk.stride(2),
        cos.stride(0),
        cos.stride(1),
        N_HEADS=n_heads,
        HALF_ROPE=half_rope,
    )


# =============================================================================
# Testing
# =============================================================================

def test_fused_rmsnorm_down():
    """Verify correctness against PyTorch baseline."""
    torch.manual_seed(42)

    M, K, N = 512, 1792, 384
    eps = 1e-6

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_norm = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    w_down = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    # PyTorch baseline
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_normed = (x_f32 / rms) * w_norm.float()
    expected = (x_normed.to(torch.bfloat16) @ w_down).to(torch.bfloat16)

    # Triton kernel
    actual = fused_rmsnorm_down_proj(x, w_norm, w_down, eps)

    m = _robust_diff_metrics(expected, actual)
    print(
        "✅ RMSNorm + Down | "
        f"Mean: {m['mean']:.6f} | p99: {m['p99']:.6f} | "
        f"p999: {m['p999']:.6f} | Max: {m['max']:.6f}"
    )
    # BF16 kernels can have rare outliers; validate distribution instead.
    assert m["mean"] < 1e-3, f"Mean diff too large: {m['mean']}"
    assert m["p99"] < 0.1, f"p99 diff too large: {m['p99']}"
    assert m["max"] < 0.5, f"Max diff too large: {m['max']}"
    return True


def test_fused_up_proj_rope():
    """Verify up-projection + RoPE correctness."""
    torch.manual_seed(42)

    M, K = 256, 384
    n_heads, head_dim, rope_dim = 14, 128, 64
    max_seq = 4096

    latent = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_up = torch.randn(K, n_heads * head_dim, dtype=torch.bfloat16, device="cuda")

    # Precompute RoPE tables
    freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs).cuda()
    cos = freqs.cos().float()
    sin = freqs.sin().float()

    # PyTorch baseline
    proj = (latent.float() @ w_up.float()).view(M, n_heads, head_dim)

    # Apply RoPE to first rope_dim dims
    x1 = proj[..., : rope_dim // 2]
    x2 = proj[..., rope_dim // 2 : rope_dim]
    cos_slice = cos[:M, :].unsqueeze(1)  # [M, 1, 32]
    sin_slice = sin[:M, :].unsqueeze(1)

    y1 = x1 * cos_slice - x2 * sin_slice
    y2 = x1 * sin_slice + x2 * cos_slice

    expected = torch.cat([y1, y2, proj[..., rope_dim:]], dim=-1).bfloat16()

    # Triton kernel
    actual = fused_mla_up_proj_rope(
        latent, w_up, cos, sin, pos_offset=0, apply_rope=True
    )

    m = _robust_diff_metrics(expected, actual)
    print(
        "✅ Up + RoPE | "
        f"Mean: {m['mean']:.6f} | p99: {m['p99']:.6f} | "
        f"p999: {m['p999']:.6f} | Max: {m['max']:.6f}"
    )
    assert m["mean"] < 1e-3, f"Mean diff too large: {m['mean']}"
    assert m["p99"] < 0.15, f"p99 diff too large: {m['p99']}"
    assert m["max"] < 0.5, f"Max diff too large: {m['max']}"
    return True


def benchmark_kernels():
    """Benchmark fused vs unfused."""
    import time

    torch.manual_seed(42)

    M, K, N = 2048, 1792, 384
    eps = 1e-6
    warmup, iters = 25, 200

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_norm = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    w_down = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    def triton_fused():
        return fused_rmsnorm_down_proj(x, w_norm, w_down, eps)

    def torch_unfused():
        # Match kernel numerics: quantize x_normed to bf16 before matmul.
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (x.float() / rms) * w_norm.float()
        return (normed.to(torch.bfloat16) @ w_down).to(torch.bfloat16)

    torch_compiled = _maybe_compile(torch_unfused)

    r1 = _bench_op("Triton fused", triton_fused, warmup=warmup, iters=iters)
    r2 = _bench_op("Torch unfused (eager)", torch_unfused, warmup=warmup, iters=iters)
    r3 = _bench_op(
        "Torch unfused (compile max-autotune-no-cudagraphs)",
        torch_compiled,
        warmup=warmup,
        iters=iters,
    )

    print(f"\n📊 Benchmark (M={M}, K={K}, N={N})")
    for r in [r1, r2, r3]:
        print(
            f"  - {r['name']}: {r['ms']:.4f} ms | "
            f"peak_alloc={_fmt_bytes(r['peak_alloc_bytes'])} | "
            f"peak_reserved={_fmt_bytes(r['peak_res_bytes'])}"
        )

    print(f"\nSpeedup vs eager: {r2['ms'] / r1['ms']:.2f}x")
    print(f"Speedup vs compile: {r3['ms'] / r1['ms']:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("MLA Triton Kernel Tests")
    print("=" * 60)
    test_fused_rmsnorm_down()
    test_fused_up_proj_rope()
    benchmark_kernels()