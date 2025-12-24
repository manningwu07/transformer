# params.py
from dataclasses import dataclass
import os
 
# --- Unified Paths ---
DATA_DIR = "data"
TOKENIZER_PATH = os.path.join(DATA_DIR, "json", "tokenizer.json")
VOCAB_PATH = os.path.join(DATA_DIR, "json", "vocab.json")
SHARD_DIR = os.path.join(DATA_DIR, "shards")

# Toggle this to switch between Training (Short Ctx) and Fine-tuning (Long Ctx)
# Options: "pretrain_5080", "longctx_5090"
MODE = "pretrain_5080" 

@dataclass
class ModelArgs:
    # 1B Model Dimensions
    d_model: int = 2048        # Core thinking dimension
    n_layers: int = 24         # Depth
    n_heads: int = 16          # Attention heads
    head_dim: int = 128        # d_model // n_heads
    
    # MLA Specifics (The Compression)
    d_latent: int = 512        # KV Cache is compressed to this size (4x smaller than d_model)
    q_lora_rank: int = 1536    # Query compression (optional, but follows DeepSeek V3 spec)
    
    # Vocabulary & Normalization
    vocab_size: int = 65535    # 2**16 - 1 to fit in uint16 (maximize tokenizer size)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # Architecture
    hidden_size: int = 5632    # MLP Expansion (~2.75x d_model for SwiGLU efficiency)
    dropout: float = 0.0
    
    # Sequence length (for generation crop)
    max_seq_len: int = 16384

@dataclass
class TrainingArgs:
    batch_size: int
    seq_len: int
    grad_accum_steps: int
    lr_start: float
    lr_end: float
    warmup_steps: int = 2000

# --- Hardware Profiles ---

if MODE == "pretrain_5080":
    # Optimized for RTX 5080 (16GB VRAM)
    Config = ModelArgs()
    TrainCfg = TrainingArgs(
        batch_size=12,          # Higher BS thanks to MLA
        seq_len=2048,           # Standard context
        grad_accum_steps=8,     # Effective batch ~128
        lr_start=3e-4,
        lr_end=1e-5,
        warmup_steps=2000
    )

elif MODE == "longctx_5090":
    # Optimized for RTX 5090 (32GB VRAM) - Pushing limits
    Config = ModelArgs()
    TrainCfg = TrainingArgs(
        batch_size=4,           # Lower BS for massive context
        seq_len=16384,          # 16k Context (MLA makes this possible!)
        grad_accum_steps=32,
        lr_start=5e-5,          # Lower LR for long-context finetuning
        lr_end=1e-6,
        warmup_steps=500
    )