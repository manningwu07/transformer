import torch
import time
import numpy as np
from bench.fused_adafactor import FusedAdafactor as LegacyAdafactor
from fused_adafactor_2pass import FusedAdafactor2Pass

# Mocking the torch.compile setup
def get_compiled_optimizer(params, lr):
    # torch.compile doesn't compile the optimizer class itself, 
    # it compiles the model. But we can simulate the 'eager' baseline 
    # vs the Triton-fused kernels.
    return torch.optim.Adafactor(params, lr=lr, relative_step=False)

def run_bench():
    device = "cuda"
    torch.cuda.set_device(0)
    
    # 1B model scale dimensions (Typical MLP/Projection size)
    # 1792 (d_model) x 4608 (hidden_size)
    M, N = 1792, 4608
    steps = 1000
    warmup = 50

    print(f"🚀 Benchmarking Adafactor at scale: {M}x{N}")

    def get_data():
        p = torch.randn((M, N), device=device, dtype=torch.bfloat16)
        g = torch.randn((M, N), device=device, dtype=torch.bfloat16)
        return p, g

    results = {}

    configs = [
        ("Torch Eager (Native)", lambda p: torch.optim.Adafactor([p], lr=1e-3)),
        ("Legacy Triton (3-Pass)", lambda p: LegacyAdafactor([p], lr=1e-3)),
        ("New Triton (2-Pass)", lambda p: FusedAdafactor2Pass([p], lr=1e-3)),
    ]

    for name, opt_factory in configs:
        p, g = get_data()
        opt = opt_factory(p)
        
        # Warmup
        for _ in range(warmup):
            p.grad = g
            opt.step()
            opt.zero_grad()
        
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1e6
        t0 = time.perf_counter()
        
        # Benchmarking Loop
        for _ in range(steps):
            p.grad = g
            opt.step()
            opt.zero_grad()
            
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        end_mem = torch.cuda.memory_allocated() / 1e6
        
        avg_time_ms = ((t1 - t0) / steps) * 1000
        mem_used = end_mem - start_mem
        results[name] = (avg_time_ms, mem_used)
        
        print(f"| {name:25} | {avg_time_ms:8.3f} ms | {mem_used:8.2f} MB State |")

    # --- THE PARITY CHECK ---
    print("\n🧐 Verifying multi-step parity (Legacy vs 2-Pass, BF16)...")
    p0, g0 = get_data()

    p_leg = p0.detach().clone()
    p_new = p0.detach().clone()

    opt_leg = LegacyAdafactor([p_leg], lr=1e-3)
    opt_new = FusedAdafactor2Pass([p_new], lr=1e-3)

    # Fixed gradient stream (same grad every step)
    for _ in range(1000):
        p_leg.grad = g0
        p_new.grad = g0
        opt_leg.step()
        opt_new.step()
        opt_leg.zero_grad(set_to_none=True)
        opt_new.zero_grad(set_to_none=True)

    diff = (p_leg - p_new).abs()
    print(
        f"legacy vs 2pass | max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f}"
    )

if __name__ == "__main__":
    run_bench()