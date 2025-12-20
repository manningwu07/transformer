import time
import psutil
import os
import mlx.core as mx
import torch
import numpy as np
import gc

# --- 1. Universal Benchmark Config ---
class BenchmarkConfig:
    # Common Params (GPT-2 Small Scale)
    d_model = 768
    n_layers = 12
    n_heads = 12
    num_heads = 12
    hidden_size = 3072  # 4 * d_model (SwiGLU)
    vocab_size = 50257
    max_len = 2048
    dropout = 0.0
    
    # New MLA / RoPE Params (Required for your new MLX code)
    head_dim = 64        # d_model / n_heads
    q_lora_rank = 128    # MLA compression rank
    d_latent = 512       # KV compression dimension
    rope_theta = 10000.0
    rms_norm_eps = 1e-6

    # Test Settings
    tokens_to_gen = 100  
    
CONFIG = BenchmarkConfig()

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# --- 2. PyTorch Benchmark (Standard Attn) ---
def benchmark_torch():
    print(f"\n{'='*10} PyTorch (Legacy) {'='*10}")
    try:
        from transformer_torch import LLM as TorchLLM
        
        print("Initializing PyTorch Model...")
        t0 = time.time()
        
        # Map config to args expected by old model
        model = TorchLLM(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            n_heads=CONFIG.num_heads,
            n_layers=CONFIG.n_layers,
            d_ff=CONFIG.hidden_size,
            max_len=CONFIG.max_len,
            dropout=CONFIG.dropout
        )
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"Init Time: {time.time() - t0:.2f}s | Device: {device}")

        # Input
        input_ids = torch.randint(0, CONFIG.vocab_size, (1, 10)).to(device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            model.generate(input_ids, max_tokens=2)
            
        # Run
        if device == "mps": torch.mps.synchronize()
        start = time.time()
        print(f"Generating {CONFIG.tokens_to_gen} tokens...")
        with torch.no_grad():
            _ = model.generate(input_ids, max_tokens=CONFIG.tokens_to_gen)
        if device == "mps": torch.mps.synchronize()
        end = time.time()

        tps = CONFIG.tokens_to_gen / (end - start)
        print(f"⚡ PyTorch Speed: {tps:.2f} tokens/sec")
        return tps

    except Exception as e:
        print(f"❌ PyTorch Error: {e}")
        return 0

# --- 3. MLX Benchmark (MLA Architecture) ---
def benchmark_mlx():
    print(f"\n{'='*10} MLX (DeepSeek MLA) {'='*10}")
    try:
        from transformer_mlx import LLM as MlxLLM
        
        print("Initializing MLX Model...")
        t0 = time.time()
        
        # Init with Config
        model = MlxLLM(config=CONFIG)
        
        # Compile & Cast to float16 (Native Speed)
        mx.eval(model.parameters())
        def to_fp16(p):
            if mx.issubdtype(p.dtype, mx.floating): return p.astype(mx.float16)
            return p
        from mlx.utils import tree_map
        model.update(tree_map(to_fp16, model.parameters()))
        
        print(f"Init Time: {time.time() - t0:.2f}s | Device: GPU (Native)")

        # Input
        input_ids = mx.random.randint(0, CONFIG.vocab_size, (1, 10))
        
        # Warmup
        print("Warming up (compiling graph)...")
        for _ in model.generate(input_ids, max_tokens=2): pass
        mx.async_eval(model.parameters())
        
        # Run
        start = time.time()
        print(f"Generating {CONFIG.tokens_to_gen} tokens...")
        
        token_count = 0
        # Iterate generator
        for _ in model.generate(input_ids, max_tokens=CONFIG.tokens_to_gen):
            token_count += 1
        
        # Force sync
        mx.eval(_) 
        end = time.time()

        tps = token_count / (end - start)
        print(f"⚡ MLX Speed: {tps:.2f} tokens/sec")
        return tps

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ MLX Error: {e}")
        return 0

# --- 4. Main Race ---
if __name__ == "__main__":
    print(f"🏁 Starting Arch Comparison")
    
    # Run PyTorch
    mem_start = get_memory_mb()
    torch_tps = benchmark_torch()
    mem_torch = get_memory_mb() - mem_start
    
    # Cleanup VRAM
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Run MLX
    mem_start = get_memory_mb()
    mlx_tps = benchmark_mlx()
    mem_mlx = get_memory_mb() - mem_start

    # --- SCOREBOARD ---
    print("\n" + "="*45)
    print(f"{'METRIC':<15} | {'PYTORCH':<15} | {'MLX (MLA)':<15}")
    print("-" * 45)
    print(f"{'TPS':<15} | {torch_tps:<15.2f} | {mlx_tps:<15.2f}")
    print(f"{'RAM Increase':<15} | {mem_torch:<15.1f} MB | {mem_mlx:<15.1f} MB")
    print("-" * 45)
    
    if torch_tps > 0:
        speedup = mlx_tps / torch_tps
        print(f"🚀 Speedup: {speedup:.2f}x")
    else:
        print("🚀 Speedup: Infinite (PyTorch failed)")