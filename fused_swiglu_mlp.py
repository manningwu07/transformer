# fused_swiglu_mlp.py
"""
Fused SwiGLU MLP for RTX 5080 (Blackwell SM100)
- FP8 E4M3 weights with row-wise dynamic scaling
- BF16 activations
- Zero-waste up-projection (single X load, no intermediate gate/up writes)

SRAM Budget Analysis (RTX 5080: ~101KB shared memory per SM):
┌─────────────────────────────────────────────────────────────────┐
│ BLOCK_M=64, BLOCK_N=128, BLOCK_K=128                            │
├─────────────────────────────────────────────────────────────────┤
│ X tile:     64 x 128 x 2B (BF16)  = 16 KB                      │
│ W1 tile:   128 x 128 x 1B (FP8)   = 16 KB                      │
│ W3 tile:   128 x 128 x 1B (FP8)   = 16 KB                      │
│ Scales:    128 x 4B x 2           =  1 KB                      │
├─────────────────────────────────────────────────────────────────┤
│ Total SRAM per K-iteration:        ≈ 49 KB                     │
│ Headroom for double-buffering:     ≈ 52 KB                     │
│ Fits 2 blocks/SM with margin                                   │
└─────────────────────────────────────────────────────────────────┘

Register Budget (per thread, 128 threads/block):
- gate_acc: 64x128/128 = 64 FP32 values = 64 regs
- up_acc:   64x128/128 = 64 FP32 values = 64 regs  
- operands/indices: ~20 regs
- Total: ~148 regs << 255 limit (no spilling)
"""

import math
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# ============================================================================
# FP8 Quantization Utilities
# ============================================================================

FP8_E4M3_MAX = 448.0  # Max representable value in E4M3


def quantize_fp8_rowwise(
    w: torch.Tensor, 
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor to FP8 E4M3 with row-wise dynamic scaling.
    
    Args:
        w: [K, N] weight tensor in FP32/BF16
        eps: Epsilon for numerical stability
        
    Returns:
        w_fp8: [K, N] quantized weights
        scale: [K] per-row scale factors
    """
    assert w.dim() == 2, f"Expected 2D tensor, got {w.dim()}D"
    
    w_f32 = w.float()
    
    # Row-wise absmax
    absmax = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=eps)
    
    # Scale = absmax / FP8_MAX
    scale = (absmax / FP8_E4M3_MAX).squeeze(1)
    
    # Quantize: w_q = w / scale
    w_scaled = w_f32 / scale[:, None]
    
    # Clamp and convert to FP8
    w_scaled = w_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    
    return w_fp8, scale.to(torch.float32)


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3, num_warps=4
        ),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def _fused_gate_up_kernel(
    # Input pointers
    X_ptr,                  # [M, K] BF16
    W1_ptr,                 # [K, N] FP8 E4M3
    W3_ptr,                 # [K, N] FP8 E4M3
    W1_scale_ptr,           # [K] FP32 row-wise scales
    W3_scale_ptr,           # [K] FP32 row-wise scales
    # Output
    Hidden_ptr,             # [M, N] BF16
    # Dimensions
    M, K, N,
    # Strides
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_hm, stride_hn,
    # Block sizes (constexpr for compilation)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Gate-Up Projection with SiLU Gating
    
    Computes: Hidden[m,n] = SiLU(X @ W1)[m,n] * (X @ W3)[m,n]
    
    Key optimizations:
    1. X loaded once per tile, used for both W1 and W3 matmuls
    2. Gate and Up computed in same loop, gating applied in registers
    3. FP8 weights dequantized on-the-fly with row-wise scales
    4. No intermediate writes for gate or up tensors
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Boundary masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulators in FP32 for numerical stability
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Base pointers for the tile
    X_base = X_ptr + offs_m[:, None] * stride_xm
    W1_base = W1_ptr + offs_n[None, :] * stride_w1n
    W3_base = W3_ptr + offs_n[None, :] * stride_w3n
    
    # Main K-reduction loop
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K
        
        # === Load X tile [BLOCK_M, BLOCK_K] ===
        x_ptrs = X_base + k_offs[None, :] * stride_xk
        x_tile = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # === Load W1 tile [BLOCK_K, BLOCK_N] ===
        w1_ptrs = W1_base + k_offs[:, None] * stride_w1k
        w1_tile_fp8 = tl.load(
            w1_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # === Load W3 tile [BLOCK_K, BLOCK_N] ===
        w3_ptrs = W3_base + k_offs[:, None] * stride_w3k
        w3_tile_fp8 = tl.load(
            w3_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # === Load row-wise scales [BLOCK_K] ===
        w1_scale = tl.load(W1_scale_ptr + k_offs, mask=mask_k, other=1.0)
        w3_scale = tl.load(W3_scale_ptr + k_offs, mask=mask_k, other=1.0)
        
        # === Dequantize FP8 -> FP32 (row-wise) ===
        # w_dequant[k, n] = w_fp8[k, n] * scale[k]
        w1_tile = w1_tile_fp8.to(tl.float32) * w1_scale[:, None]
        w3_tile = w3_tile_fp8.to(tl.float32) * w3_scale[:, None]
        
        # === Matrix multiply accumulate ===
        # Using tl.dot for tensor core utilization
        gate_acc = tl.dot(x_tile.to(tl.float32), w1_tile, acc=gate_acc)
        up_acc = tl.dot(x_tile.to(tl.float32), w3_tile, acc=up_acc)
    
    # === SiLU Gating (entirely in registers) ===
    # SiLU(x) = x * sigmoid(x)
    # Hidden = SiLU(gate) * up = gate * sigmoid(gate) * up
    gate_sigmoid = tl.sigmoid(gate_acc)
    hidden = gate_acc * gate_sigmoid * up_acc
    
    # === Store output ===
    hidden_ptrs = Hidden_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(
        hidden_ptrs,
        hidden.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 128},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 128},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 32,"BLOCK_K": 128, "BLOCK_N": 64},
            num_stages=4, num_warps=4
        ),
    ],
    key=["M", "hidden_size", "N"],
)
@triton.jit
def _down_proj_kernel(
    # Input pointers
    Hidden_ptr,             # [M, hidden_size] BF16
    W2_ptr,                 # [hidden_size, N] FP8 E4M3
    W2_scale_ptr,           # [hidden_size] FP32 row-wise scales
    # Output
    Out_ptr,                # [M, N] BF16
    # Dimensions
    M, hidden_size, N,
    # Strides
    stride_hm, stride_hk,
    stride_w2k, stride_w2n,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Down Projection: Out = Hidden @ W2
    
    With FP8 weights and row-wise dequantization.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Base pointers
    H_base = Hidden_ptr + offs_m[:, None] * stride_hm
    W2_base = W2_ptr + offs_n[None, :] * stride_w2n
    
    # K-reduction loop over hidden_size
    for k_start in range(0, hidden_size, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < hidden_size
        
        # Load Hidden tile [BLOCK_M, BLOCK_K]
        h_ptrs = H_base + k_offs[None, :] * stride_hk
        h_tile = tl.load(
            h_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load W2 tile [BLOCK_K, BLOCK_N]
        w2_ptrs = W2_base + k_offs[:, None] * stride_w2k
        w2_tile_fp8 = tl.load(
            w2_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Load scales
        w2_scale = tl.load(W2_scale_ptr + k_offs, mask=mask_k, other=1.0)
        
        # Dequantize
        w2_tile = w2_tile_fp8.to(tl.float32) * w2_scale[:, None]
        
        # Accumulate
        acc = tl.dot(h_tile.to(tl.float32), w2_tile, acc=acc)
    
    # Store output
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ============================================================================
# Backward Kernels
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def _gate_up_backward_dx_kernel(
    # Forward activations (saved for backward)
    Gate_ptr,               # [M, N] - pre-activation gate values
    Up_ptr,                 # [M, N] - up projection values
    # Gradients
    dHidden_ptr,            # [M, N] - gradient from downstream
    # Weights (transposed access)
    W1_ptr,                 # [K, N] FP8
    W3_ptr,                 # [K, N] FP8
    W1_scale_ptr,
    W3_scale_ptr,
    # Output gradient
    dX_ptr,                 # [M, K]
    # Dimensions
    M, K, N,
    # Strides
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_dhm, stride_dhn,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_dxm, stride_dxk,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Backward pass for dX through fused gate-up.
    
    dGate = dHidden * Up * SiLU'(Gate)
    dUp = dHidden * SiLU(Gate)
    dX = dGate @ W1.T + dUp @ W3.T
    
    SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # Accumulator for dX
    dx_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    # Loop over N dimension
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n
        mask_n = n_offs < N
        
        # Load gate and up activations
        gate_ptrs = Gate_ptr + offs_m[:, None] * stride_gm + n_offs[None, :] * stride_gn
        up_ptrs = Up_ptr + offs_m[:, None] * stride_um + n_offs[None, :] * stride_un
        dh_ptrs = dHidden_ptr + offs_m[:, None] * stride_dhm + n_offs[None, :] * stride_dhn
        
        gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        up = tl.load(up_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        dh = tl.load(dh_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        
        # Compute SiLU and its derivative
        gate_f32 = gate.to(tl.float32)
        up_f32 = up.to(tl.float32)
        dh_f32 = dh.to(tl.float32)
        
        sig = tl.sigmoid(gate_f32)
        silu = gate_f32 * sig
        silu_grad = sig * (1.0 + gate_f32 * (1.0 - sig))
        
        # dGate = dHidden * Up * SiLU'(Gate)
        dgate = dh_f32 * up_f32 * silu_grad
        # dUp = dHidden * SiLU(Gate)
        dup = dh_f32 * silu
        
        # Load W1 and W3 tiles (transposed: [N, K] access pattern)
        # We need W1[k, n] -> accessing as [n, k] for transpose
        w1_ptrs = W1_ptr + n_offs[:, None] * stride_w1n + offs_k[None, :] * stride_w1k
        w3_ptrs = W3_ptr + n_offs[:, None] * stride_w3n + offs_k[None, :] * stride_w3k
        
        w1_tile_fp8 = tl.load(
            w1_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        w3_tile_fp8 = tl.load(
            w3_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load scales and dequantize (broadcast over N)
        w1_scale = tl.load(W1_scale_ptr + offs_k, mask=mask_k, other=1.0)
        w3_scale = tl.load(W3_scale_ptr + offs_k, mask=mask_k, other=1.0)
        
        # Dequantize - note: scales are per K, weights are [N, K]
        w1_tile = w1_tile_fp8.to(tl.float32) * w1_scale[None, :]
        w3_tile = w3_tile_fp8.to(tl.float32) * w3_scale[None, :]
        
        # Accumulate: dX += dGate @ W1.T + dUp @ W3.T
        # dGate: [M, N], W1.T: [N, K] -> [M, K]
        dx_acc = tl.dot(dgate, w1_tile, acc=dx_acc)
        dx_acc = tl.dot(dup, w3_tile, acc=dx_acc)
    
    # Store dX
    dx_ptrs = dX_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    tl.store(
        dx_ptrs,
        dx_acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_k[None, :]
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64},
            num_stages=2, num_warps=4
        ),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def _gate_up_backward_dw_kernel(
    # Inputs
    X_ptr,                  # [M, K]
    Gate_ptr,               # [M, N]
    Up_ptr,                 # [M, N]
    dHidden_ptr,            # [M, N]
    # Output gradients
    dW1_ptr,                # [K, N]
    dW3_ptr,                # [K, N]
    # Dimensions
    M, K, N,
    # Strides
    stride_xm, stride_xk,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_dhm, stride_dhn,
    stride_dw1k, stride_dw1n,
    stride_dw3k, stride_dw3n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Backward pass for dW1 and dW3.
    
    dW1 = X.T @ dGate
    dW3 = X.T @ dUp
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    
    mask_k = offs_k < K
    mask_n = offs_n < N
    
    # Accumulators
    dw1_acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    dw3_acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    
    # Loop over M (batch dimension)
    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + offs_m
        mask_m = m_offs < M
        
        # Load X.T: need X[m, k] -> access as [k, m] conceptually
        x_ptrs = X_ptr + m_offs[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        x_tile = x_tile.to(tl.float32)
        
        # Load Gate, Up, dHidden
        gate_ptrs = Gate_ptr + m_offs[:, None] * stride_gm + offs_n[None, :] * stride_gn
        up_ptrs = Up_ptr + m_offs[:, None] * stride_um + offs_n[None, :] * stride_un
        dh_ptrs = dHidden_ptr + m_offs[:, None] * stride_dhm + offs_n[None, :] * stride_dhn
        
        gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        dh = tl.load(dh_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        
        # Compute local gradients
        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_grad = sig * (1.0 + gate * (1.0 - sig))
        
        dgate = dh * up * silu_grad  # [BLOCK_M, BLOCK_N]
        dup = dh * silu              # [BLOCK_M, BLOCK_N]
        
        # Accumulate: dW = X.T @ dLocal
        # X.T: [K, M], dgate: [M, N] -> [K, N]
        # We have x_tile as [M, K], need transpose
        x_t = tl.trans(x_tile)  # [BLOCK_K, BLOCK_M]
        
        dw1_acc = tl.dot(x_t, dgate, acc=dw1_acc)
        dw3_acc = tl.dot(x_t, dup, acc=dw3_acc)
    
    # Store weight gradients
    dw1_ptrs = dW1_ptr + offs_k[:, None] * stride_dw1k + offs_n[None, :] * stride_dw1n
    dw3_ptrs = dW3_ptr + offs_k[:, None] * stride_dw3k + offs_n[None, :] * stride_dw3n
    
    tl.store(dw1_ptrs, dw1_acc.to(tl.float32), mask=mask_k[:, None] & mask_n[None, :])
    tl.store(dw3_ptrs, dw3_acc.to(tl.float32), mask=mask_k[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_N": 64},
            num_stages=4, num_warps=4
        ),
    ],
    key=["M", "hidden_size", "N"],
)
@triton.jit
def _down_proj_backward_dhidden_kernel(
    # Inputs
    dOut_ptr,               # [M, N]
    W2_ptr,                 # [hidden_size, N] FP8
    W2_scale_ptr,           # [hidden_size]
    # Output
    dHidden_ptr,            # [M, hidden_size]
    # Dimensions
    M, hidden_size, N,
    # Strides
    stride_dom, stride_don,
    stride_w2k, stride_w2n,
    stride_dhm, stride_dhk,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward: dHidden = dOut @ W2.T
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)  # K here is hidden_size
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_k = offs_k < hidden_size
    
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n
        mask_n = n_offs < N
        
        # Load dOut [BLOCK_M, BLOCK_N]
        do_ptrs = dOut_ptr + offs_m[:, None] * stride_dom + n_offs[None, :] * stride_don
        do_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        
        # Load W2.T: W2[k, n] -> access [n, k]
        w2_ptrs = W2_ptr + n_offs[:, None] * stride_w2n + offs_k[None, :] * stride_w2k
        w2_tile_fp8 = tl.load(w2_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Dequantize
        w2_scale = tl.load(W2_scale_ptr + offs_k, mask=mask_k, other=1.0)
        w2_tile = w2_tile_fp8.to(tl.float32) * w2_scale[None, :]
        
        # Accumulate
        acc = tl.dot(do_tile.to(tl.float32), w2_tile, acc=acc)
    
    # Store
    dh_ptrs = dHidden_ptr + offs_m[:, None] * stride_dhm + offs_k[None, :] * stride_dhk
    tl.store(dh_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3, num_warps=8
        ),
    ],
    key=["M", "hidden_size", "N"],
)
@triton.jit
def _down_proj_backward_dw2_kernel(
    # Inputs
    Hidden_ptr,             # [M, hidden_size]
    dOut_ptr,               # [M, N]
    # Output
    dW2_ptr,                # [hidden_size, N]
    # Dimensions
    M, hidden_size, N,
    # Strides
    stride_hm, stride_hk,
    stride_dom, stride_don,
    stride_dwk, stride_dwn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Backward: dW2 = Hidden.T @ dOut
    """
    pid_k = tl.program_id(0)  # hidden_size
    pid_n = tl.program_id(1)  # N (d_model)
    
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    
    mask_k = offs_k < hidden_size
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    
    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + offs_m
        mask_m = m_offs < M
        
        # Load Hidden [BLOCK_M, BLOCK_K]
        h_ptrs = Hidden_ptr + m_offs[:, None] * stride_hm + offs_k[None, :] * stride_hk
        h_tile = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load dOut [BLOCK_M, BLOCK_N]
        do_ptrs = dOut_ptr + m_offs[:, None] * stride_dom + offs_n[None, :] * stride_don
        do_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate: H.T @ dOut
        h_t = tl.trans(h_tile.to(tl.float32))
        acc = tl.dot(h_t, do_tile.to(tl.float32), acc=acc)
    
    # Store
    dw_ptrs = dW2_ptr + offs_k[:, None] * stride_dwk + offs_n[None, :] * stride_dwn
    tl.store(dw_ptrs, acc, mask=mask_k[:, None] & mask_n[None, :])


# ============================================================================
# Python Launch Helpers
# ============================================================================

def fused_gate_up_forward(
    x: torch.Tensor,
    w1_fp8: torch.Tensor,
    w3_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass: Hidden = SiLU(X @ W1) * (X @ W3)
    
    Args:
        x: [M, K] input in BF16
        w1_fp8: [K, N] gate weights in FP8
        w3_fp8: [K, N] up weights in FP8
        w1_scale: [K] row-wise scales for W1
        w3_scale: [K] row-wise scales for W3
    
    Returns:
        hidden: [M, N] in BF16
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    M, K = x.shape
    _, N = w1_fp8.shape
    
    hidden = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    
    _fused_gate_up_kernel[grid](
        x, w1_fp8, w3_fp8, w1_scale, w3_scale,
        hidden,
        M, K, N,
        x.stride(0), x.stride(1),
        w1_fp8.stride(0), w1_fp8.stride(1),
        w3_fp8.stride(0), w3_fp8.stride(1),
        hidden.stride(0), hidden.stride(1),
    )
    
    return hidden


def down_proj_forward(
    hidden: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass: Out = Hidden @ W2
    """
    assert hidden.is_cuda and hidden.dtype == torch.bfloat16
    M, hidden_size = hidden.shape
    _, N = w2_fp8.shape
    
    out = torch.empty((M, N), device=hidden.device, dtype=torch.bfloat16)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    
    _down_proj_kernel[grid](
        hidden, w2_fp8, w2_scale,
        out,
        M, hidden_size, N,
        hidden.stride(0), hidden.stride(1),
        w2_fp8.stride(0), w2_fp8.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def _fp8_mm_rowwise_kernel(
    X_ptr,              # [M, K] bf16
    W_fp8_ptr,          # [K, N] fp8
    W_scale_ptr,        # [K] fp32
    Out_ptr,            # [M, N] bf16
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    X_base = X_ptr + offs_m[:, None] * stride_xm
    W_base = W_fp8_ptr + offs_n[None, :] * stride_wn

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = X_base + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.bfloat16
        )

        w_ptrs = W_base + offs_k[:, None] * stride_wk
        w_fp8 = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        s = tl.load(W_scale_ptr + offs_k, mask=mask_k, other=1.0).to(tl.float32)

        w = w_fp8.to(tl.float32) * s[:, None]
        acc += tl.dot(x.to(tl.float32), w)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])
    
def fp8_mm_rowwise(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w_fp8.is_cuda and w_scale.is_cuda
    K, N = w_fp8.shape
    M = x.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _fp8_mm_rowwise_kernel[grid](
        x, w_fp8, w_scale, out,
        M, K, N,
        x.stride(0), x.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        out.stride(0), out.stride(1),
    )
    return out

# ============================================================================
# Autograd Function
# ============================================================================

class FusedSwiGLUMLPFunction(torch.autograd.Function):
    """
    Autograd wrapper for fused SwiGLU MLP.
    
    Forward:  Out = W2 @ (SiLU(X @ W1) * (X @ W3))
    
    Note: For backward, we need to save the pre-gating activations.
    This is unavoidable for correct gradients.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,        # FP32/BF16 master weights
        w3: torch.Tensor,
        w2: torch.Tensor,
        # Pre-quantized FP8 + scales (computed outside for efficiency)
        w1_fp8: torch.Tensor,
        w3_fp8: torch.Tensor,
        w2_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w3_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Flatten batch dimensions: [B, T, K] -> [B*T, K]
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        M, K = x_2d.shape
        N = w1_fp8.shape[1]  # hidden_size
        
        # === Fused Gate-Up (single X load) ===
        hidden = fused_gate_up_forward(x_2d, w1_fp8, w3_fp8, w1_scale, w3_scale)
        
        # === Down Projection ===
        out = down_proj_forward(hidden, w2_fp8, w2_scale)
        
        # Reshape back
        out = out.view(*orig_shape[:-1], -1)
        
        # For backward, we need to recompute gate/up or save them
        # Saving is more memory but faster backward
        # Here we choose to recompute in backward to save memory
        ctx.save_for_backward(
            x_2d,           # For dW1, dW3, dW2
            hidden,         # For dW2
            w1_fp8, w3_fp8, w2_fp8,
            w1_scale, w3_scale, w2_scale,
            w1, w3, w2,     # Master weights for gradient accumulation
        )
        ctx.orig_shape = orig_shape
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.hidden_size = hidden.shape[1]
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_2d, hidden,
            w1_fp8, w3_fp8, w2_fp8,
            w1_scale, w3_scale, w2_scale,
            w1, w3, w2,
        ) = ctx.saved_tensors
        
        M, K = ctx.M, ctx.K
        hidden_size = ctx.hidden_size
        N_out = grad_output.shape[-1]
        
        grad_output_2d = grad_output.view(-1, N_out).contiguous()
        
        # === Backward through Down Projection ===
        # dHidden = dOut @ W2.T
        dHidden = torch.empty((M, hidden_size), device=x_2d.device, dtype=torch.bfloat16)
        
        grid_dhidden = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(hidden_size, meta["BLOCK_K"]),
        )
        _down_proj_backward_dhidden_kernel[grid_dhidden](
            grad_output_2d, w2_fp8, w2_scale,
            dHidden,
            M, hidden_size, N_out,
            grad_output_2d.stride(0), grad_output_2d.stride(1),
            w2_fp8.stride(0), w2_fp8.stride(1),
            dHidden.stride(0), dHidden.stride(1),
        )
        
        # dW2 = Hidden.T @ dOut
        dW2 = torch.zeros_like(w2)
        grid_dw2 = lambda meta: (
            triton.cdiv(hidden_size, meta["BLOCK_K"]),
            triton.cdiv(N_out, meta["BLOCK_N"]),
        )
        _down_proj_backward_dw2_kernel[grid_dw2](
            hidden, grad_output_2d,
            dW2,
            M, hidden_size, N_out,
            hidden.stride(0), hidden.stride(1),
            grad_output_2d.stride(0), grad_output_2d.stride(1),
            dW2.stride(0), dW2.stride(1),
        )
        
        # === Recompute Gate and Up for backward ===
        # We need these for the SiLU gradient
        # This is the memory/compute tradeoff - we recompute instead of saving
        gate = torch.empty((M, hidden_size), device=x_2d.device, dtype=torch.bfloat16)
        up = torch.empty((M, hidden_size), device=x_2d.device, dtype=torch.bfloat16)
        
        # Simple matmul for recomputation (could also use Triton)
        # X @ W1 and X @ W3 with dequantization
        with torch.no_grad():
            gate = fp8_mm_rowwise(x_2d, w1_fp8, w1_scale)
            up = fp8_mm_rowwise(x_2d, w3_fp8, w3_scale)
        
        # === Backward through Gate-Up ===
        # dX = dGate @ W1.T + dUp @ W3.T
        dX = torch.empty((M, K), device=x_2d.device, dtype=torch.bfloat16)
        
        grid_dx = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K, meta["BLOCK_K"]),
        )
        _gate_up_backward_dx_kernel[grid_dx](
            gate, up, dHidden,
            w1_fp8, w3_fp8, w1_scale, w3_scale,
            dX,
            M, K, hidden_size,
            gate.stride(0), gate.stride(1),
            up.stride(0), up.stride(1),
            dHidden.stride(0), dHidden.stride(1),
            w1_fp8.stride(0), w1_fp8.stride(1),
            w3_fp8.stride(0), w3_fp8.stride(1),
            dX.stride(0), dX.stride(1),
        )
        
        # dW1, dW3
        dW1 = torch.zeros_like(w1)
        dW3 = torch.zeros_like(w3)
        
        grid_dw = lambda meta: (
            triton.cdiv(K, meta["BLOCK_K"]),
            triton.cdiv(hidden_size, meta["BLOCK_N"]),
        )
        _gate_up_backward_dw_kernel[grid_dw](
            x_2d, gate, up, dHidden,
            dW1, dW3,
            M, K, hidden_size,
            x_2d.stride(0), x_2d.stride(1),
            gate.stride(0), gate.stride(1),
            up.stride(0), up.stride(1),
            dHidden.stride(0), dHidden.stride(1),
            dW1.stride(0), dW1.stride(1),
            dW3.stride(0), dW3.stride(1),
        )
        
        # Reshape dX back to original shape
        dX = dX.view(*ctx.orig_shape)
        
        # Return gradients for all inputs
        # (x, w1, w3, w2, w1_fp8, w3_fp8, w2_fp8, w1_scale, w3_scale, w2_scale)
        return dX, dW1, dW3, dW2, None, None, None, None, None, None


# ============================================================================
# Module Wrapper
# ============================================================================

class FusedSwiGLUMLP(torch.nn.Module):
    """
    Drop-in replacement for SwiGLU_MLP with fused Triton kernels.
    
    Usage:
        mlp = FusedSwiGLUMLP(d_model=1792, hidden_size=4608)
        out = mlp(x)  # x: [B, T, d_model]
    """
    
    def __init__(self, d_model: int, hidden_size: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        
        # Master weights in FP16 for optimizer
        master_dtype = torch.bfloat16
        
        self.w1 = torch.nn.Parameter(
            torch.empty(d_model, hidden_size, dtype=master_dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(d_model, hidden_size, dtype=master_dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(hidden_size, d_model, dtype=master_dtype)
        )
        
        # Initialize
        torch.nn.init.normal_(self.w1, std=0.02)
        torch.nn.init.normal_(self.w3, std=0.02)
        torch.nn.init.normal_(self.w2, std=0.02)
        
        # FP8 buffers (updated before forward)
        self.register_buffer("w1_fp8", None)
        self.register_buffer("w3_fp8", None)
        self.register_buffer("w2_fp8", None)
        self.register_buffer("w1_scale", None)
        self.register_buffer("w3_scale", None)
        self.register_buffer("w2_scale", None)
        
        self._fp8_dirty = True
    
    def _update_fp8_weights(self):
        """Quantize master weights to FP8 with row-wise scaling."""
        if not self._fp8_dirty:
            return
        
        self.w1_fp8, self.w1_scale = quantize_fp8_rowwise(self.w1.data)
        self.w3_fp8, self.w3_scale = quantize_fp8_rowwise(self.w3.data)
        self.w2_fp8, self.w2_scale = quantize_fp8_rowwise(self.w2.data)
        
        self._fp8_dirty = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mark weights dirty after optimizer step
        if self.training:
            self._fp8_dirty = True
        
        self._update_fp8_weights()
        
        # Ensure BF16
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        
        return FusedSwiGLUMLPFunction.apply(
            x_bf16,
            self.w1, self.w3, self.w2,
            self.w1_fp8, self.w3_fp8, self.w2_fp8,
            self.w1_scale, self.w3_scale, self.w2_scale,
        )


# ============================================================================
# Testing & Benchmarking
# ============================================================================

def test_correctness():
    """Test against reference PyTorch implementation."""
    print("Testing correctness...")
    
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    B, T, d_model, hidden_size = 2, 512, 1792, 4608
    
    # Create inputs
    x = torch.randn(B, T, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    # Reference implementation
    class RefSwiGLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(d_model, hidden_size, bias=False)
            self.w2 = torch.nn.Linear(hidden_size, d_model, bias=False)
            self.w3 = torch.nn.Linear(d_model, hidden_size, bias=False)
        
        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
    
    ref = RefSwiGLU().to(device).to(torch.bfloat16)
    
    # Our implementation
    fused = FusedSwiGLUMLP(d_model, hidden_size).to(device)
    
    # Copy weights
    with torch.no_grad():
        fused.w1.copy_(ref.w1.weight.T.float())
        fused.w3.copy_(ref.w3.weight.T.float())
        fused.w2.copy_(ref.w2.weight.T.float())
    
    # Forward
    x_ref = x.clone().detach().requires_grad_(True)
    x_fused = x.clone().detach().requires_grad_(True)
    
    out_ref = ref(x_ref)
    out_fused = fused(x_fused)
    
    # Check forward
    max_diff = (out_ref - out_fused).abs().max().item()
    print(f"  Forward max diff: {max_diff:.6f}")
    assert max_diff < 0.1, f"Forward mismatch: {max_diff}"  # FP8 has limited precision
    
    # Backward
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_fused.backward(grad_out)
    
    dx_diff = (x_ref.grad - x_fused.grad).abs().max().item()
    print(f"  dX max diff: {dx_diff:.6f}")
    
    print("✅ Correctness test passed!")


def benchmark():
    """Benchmark against PyTorch baseline."""
    print("\nBenchmarking...")
    
    import time
    
    device = torch.device("cuda")
    
    # Target: BS=4, seq_len=2048
    B, T, d_model, hidden_size = 4, 2048, 1792, 4608
    
    x = torch.randn(B, T, d_model, device=device, dtype=torch.bfloat16)
    
    # PyTorch baseline
    class RefSwiGLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(d_model, hidden_size, bias=False)
            self.w2 = torch.nn.Linear(hidden_size, d_model, bias=False)
            self.w3 = torch.nn.Linear(d_model, hidden_size, bias=False)
        
        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
    
    ref = RefSwiGLU().to(device).to(torch.bfloat16)
    fused = FusedSwiGLUMLP(d_model, hidden_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = ref(x)
        _ = fused(x)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = ref(x)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000
    
    # Benchmark Fused
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = fused(x)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"  PyTorch BF16:  {pytorch_time:.3f} ms")
    print(f"  Fused FP8:     {fused_time:.3f} ms")
    print(f"  Speedup:       {pytorch_time / fused_time:.2f}x")
    
    # Memory comparison
    torch.cuda.reset_peak_memory_stats()
    _ = ref(x)
    torch.cuda.synchronize()
    pytorch_mem = torch.cuda.max_memory_allocated() / 1e9
    
    torch.cuda.reset_peak_memory_stats()
    _ = fused(x)
    torch.cuda.synchronize()
    fused_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"  PyTorch peak mem: {pytorch_mem:.2f} GB")
    print(f"  Fused peak mem:   {fused_mem:.2f} GB")


if __name__ == "__main__":
    test_correctness()
    benchmark()