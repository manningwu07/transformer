# params.py
from dataclasses import dataclass
import os

DATA_DIR = "data"
TOKENIZER_PATH = os.path.join(DATA_DIR, "json", "tokenizer.json")
VOCAB_PATH = os.path.join(DATA_DIR, "json", "vocab.json")
SHARD_DIR = os.path.join(DATA_DIR, "shards")

MODE = "pretrain_5080"  # or "longctx_5090"


@dataclass
class ModelArgs:
    # Alignment-first core (Maximize GPU utilization by making powers of 2 for d_model + head_dim)
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    head_dim: int = 128 

    # MLA compression
    d_latent: int = 384
    q_lora_rank: int = 512

    # Vocab & norms
    vocab_size: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MLP
    hidden_size: int = 3584
    dropout: float = 0.0

    # Runtime toggles
    max_seq_len: int = 12288
    gradient_checkpointing: bool = True
    checkpoint_every_n: int = 1
    compile_layers: bool = False

    seed: int = 1337


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