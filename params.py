# params.py
from dataclasses import dataclass
import os
from typing import Literal

DATA_DIR = "data"
TOKENIZER_PATH = os.path.join(DATA_DIR, "json", "tokenizer.json")
VOCAB_PATH = os.path.join(DATA_DIR, "json", "vocab.json")
SHARD_DIR = os.path.join(DATA_DIR, "shards")

MODE = "longctx_5090"  # or "longctx_5090"


@dataclass
class ModelArgs:
    # Alignment-first core (Maximize GPU utilization by making powers of 2 for d_model + head_dim)
    d_model: int = 1792
    n_layers: int = 28
    n_heads: int = 14
    head_dim: int = 128 

    # MLA compression
    d_latent: int = 384
    q_lora_rank = d_latent

    # Vocab & norms
    vocab_size: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MLP
    hidden_size: int = 4608 # ~x2.5 d_model
    dropout: float = 0.0

    # Runtime toggles
    max_seq_len: int = 8192
    gradient_checkpointing: bool = False # activation checkpointiny is skipped
    checkpoint_skip_every_n: int = 1 # 0 = no skipping, 1 = skip all, N = skip all but every Nth
    use_float8: bool = True
    
    # Compilation options
    compile_mode: str = "model"  # "none" | "model"
    
    # Experimental: compile autograd dispatcher (gradient accumulation, etc.)
    use_compiled_autograd: bool = False

    seed: int = 1337
    
    # Float8 (torchao)
    # Converts Linear layers inside Transformer blocks to Float8Linear for training
    # Keep embeddings + lm_head in BF16 (weight-tying + stability).
    use_float8: bool = True
    # Common recipe names in torchao docs: "rowwise" or "tensorwise"
    float8_recipe: Literal["rowwise", "tensorwise"] = "tensorwise"


@dataclass
class TrainingArgs:
    batch_size: int
    seq_len: int
    grad_accum_steps: int
    lr_start: float
    lr_end: float
    warmup_steps: int = 2000


if MODE == "pretrain_5080":
    Config = ModelArgs()
    TrainCfg = TrainingArgs(
        batch_size=2,
        seq_len=2048,
        grad_accum_steps=256,
        lr_start=1e-3,
        lr_end=1e-4,
        warmup_steps=100,
    )

elif MODE == "longctx_5090":
    Config = ModelArgs()
    TrainCfg = TrainingArgs(
        batch_size=1,
        seq_len=8192,
        grad_accum_steps=32,
        lr_start=1e-4,
        lr_end=1e-5,
        warmup_steps=100,
    )