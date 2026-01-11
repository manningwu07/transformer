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
    steps = 100
    warmup = 20

    print(f"🚀 Benchmarking Adafactor at scale: {M}x{N}")

    def get_data():
        p = torch.randn((M, N), device=device, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn((M, N), device=device, dtype=torch.bfloat16)
        return p, g

    results = {}

    configs = [
        ("Torch Eager (Native)", lambda p: torch.optim.Adafactor([p], lr=1e-3, relative_step=False)),
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
    print("\n🧐 Verifying Numerical Parity (BF16)...")
    p_ref, g_ref = get_data()
    p_new, _ = p_ref.clone(), g_ref.clone()
    p_new.grad = g_ref
    p_ref.grad = g_ref

    opt_ref = torch.optim.Adafactor([p_ref], lr=1e-3, relative_step=False)
    opt_new = FusedAdafactor2Pass([p_new], lr=1e-3)

    opt_ref.step()
    opt_new.step()

    diff = (p_ref - p_new).abs().max().item()
    if diff < 1e-3:
        print(f"✅ SUCCESS: New Triton matches Native Adafactor (Max Diff: {diff:.6f})")
    else:
        print(f"⚠️ WARNING: Divergence detected (Max Diff: {diff:.6f})")

if __name__ == "__main__":
    run_bench()