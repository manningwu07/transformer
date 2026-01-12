# bench_mlp.py
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_swiglu_mlp import FusedSwiGLUMLP


@dataclass
class BenchCfg:
    B: int = 2
    T: int = 2048
    d_model: int = 1792
    hidden: int = 4608
    iters: int = 50
    warmup: int = 10
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    compile_mode: str = "max-autotune-no-cudagraphs"  # or "default"


class RefSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def clone_weights_ref_to_fused(ref: RefSwiGLU, fused: FusedSwiGLUMLP) -> None:
    # Ref uses [hidden, d_model] weight, fused uses [d_model, hidden]
    with torch.no_grad():
        fused.w1.copy_(ref.w1.weight.T.to(fused.w1.dtype))
        fused.w3.copy_(ref.w3.weight.T.to(fused.w3.dtype))
        fused.w2.copy_(ref.w2.weight.T.to(fused.w2.dtype))
        fused._fp8_dirty = True

def time_one(
    name: str,
    fn: Callable[[], torch.Tensor],
    cfg: BenchCfg,
) -> Dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(cfg.warmup):
        y = fn()
        y.backward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        y = fn()
        y.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000.0 / cfg.iters
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    max_reserved = torch.cuda.max_memory_reserved() / 1e9

    return {
        "ms_per_iter": ms,
        "max_alloc_GB": max_alloc,
        "max_reserved_GB": max_reserved,
    }


def maybe_compile(m: nn.Module, cfg: BenchCfg) -> nn.Module:
    return torch.compile(
        m,
        mode=cfg.compile_mode,
        fullgraph=False,
        dynamic=False,
    )


def main():
    cfg = BenchCfg(
        B=int(os.getenv("B", "2")),
        T=int(os.getenv("T", "2048")),
        d_model=int(os.getenv("D", "1792")),
        hidden=int(os.getenv("H", "4608")),
        iters=int(os.getenv("ITERS", "50")),
        warmup=int(os.getenv("WARMUP", "10")),
        compile_mode=os.getenv("COMPILE_MODE", "max-autotune-no-cudagraphs"),
    )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(cfg.device)

    # Input and grad
    x = torch.randn(cfg.B, cfg.T, cfg.d_model, device=device, dtype=cfg.dtype)
    x.requires_grad_(True)

    def make_loss(y: torch.Tensor) -> torch.Tensor:
        # Simple scalar loss that touches all outputs
        return y.float().mean()

    # Build models
    ref = RefSwiGLU(cfg.d_model, cfg.hidden).to(device).to(cfg.dtype).train()
    fused = FusedSwiGLUMLP(cfg.d_model, cfg.hidden).to(device).train()

    # Align weights so outputs are comparable-ish
    clone_weights_ref_to_fused(ref, fused)

    # Reference eager
    def ref_step() -> torch.Tensor:
        ref.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(device_type="cuda", dtype=cfg.dtype):
            y = ref(x)
            loss = make_loss(y)
        return loss

    # Fused eager
    def fused_step() -> torch.Tensor:
        fused.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(device_type="cuda", dtype=cfg.dtype):
            y = fused(x)
            loss = make_loss(y)
        return loss

    # Compile models (optional)
    ref_c = maybe_compile(ref, cfg)
    fused_c = maybe_compile(fused, cfg)

    def ref_c_step() -> torch.Tensor:
        ref_c.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(device_type="cuda", dtype=cfg.dtype):
            y = ref_c(x)
            loss = make_loss(y)
        return loss

    def fused_c_step() -> torch.Tensor:
        fused_c.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(device_type="cuda", dtype=cfg.dtype):
            y = fused_c(x)
            loss = make_loss(y)
        return loss

    print(
        f"Bench: B={cfg.B} T={cfg.T} D={cfg.d_model} H={cfg.hidden} "
        f"iters={cfg.iters} warmup={cfg.warmup} dtype={cfg.dtype}"
    )
    print(
        f"Env: FUSED_MLP_SAVE_GATE_UP={os.getenv('FUSED_MLP_SAVE_GATE_UP','0')} "
        f"COMPILE_MODE={cfg.compile_mode}"
    )

    results = {}
    results["ref_eager"] = time_one("ref_eager", ref_step, cfg)
    results["ref_compile"] = time_one("ref_compile", ref_c_step, cfg)
    results["fused_eager"] = time_one("fused_eager", fused_step, cfg)
    results["fused_compile"] = time_one("fused_compile", fused_c_step, cfg)

    print("\nResults:")
    for k, v in results.items():
        print(
            f"{k:13s} | {v['ms_per_iter']:.3f} ms/iter | "
            f"alloc {v['max_alloc_GB']:.2f} GB | "
            f"reserved {v['max_reserved_GB']:.2f} GB"
        )


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()