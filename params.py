# params.py
from dataclasses import dataclass
import os
from typing import Literal

DATA_DIR = "data"
TOKENIZER_PATH = os.path.join(DATA_DIR, "json", "tokenizer.json")
VOCAB_PATH = os.path.join(DATA_DIR, "json", "vocab.json")
SHARD_DIR = os.path.join(DATA_DIR, "shards")

MODE = "pretrain_5080"  # or "longctx_5090"


@dataclass
class ModelArgs:
    # Alignment-first core (Maximize GPU utilization by making powers of 2 for d_model + head_dim)
    d_model: int = 1792
    n_layers: int = 28
    n_heads: int = 14
    head_dim: int = 128 

    # MLA compression
    d_latent: int = 320
    q_lora_rank = d_latent

    # Vocab & norms
    vocab_size: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MLP
    hidden_size: int = 4928 # x2.75 d_model
    dropout: float = 0.0

    # Runtime toggles
    max_seq_len: int = 8192
    gradient_checkpointing: bool = True
    checkpoint_skip_every_n: int = 4 # 0 = no skipping, 1 = skip all, N = skip all but every Nth
    compile_layers: bool = True  # ← Enable this now
    use_float8: bool = True

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
        batch_size=1,
        seq_len=2048,
        grad_accum_steps=256,
        lr_start=3e-4,
        lr_end=1e-5,
        warmup_steps=2000,
    )

elif MODE == "longctx_5090":
    Config = ModelArgs()
    TrainCfg = TrainingArgs(
        batch_size=1,
        seq_len=8192,
        grad_accum_steps=32,
        lr_start=5e-5,
        lr_end=1e-6,
        warmup_steps=500,
    )