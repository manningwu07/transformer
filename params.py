# params.py
from dataclasses import dataclass

# Choose a profile by setting ACTIVE = "local" / "debug" / "lora"
ACTIVE = "local"

@dataclass
class BaseConfig:
    # Core transformer parameters (must match any loaded checkpoint)
    d_model: int = 768         # model width
    hidden_size: int = 2048    # MLP hidden dim (typically 2-4x d_model)
    vocab_size: int = 65_536   # |V|
    num_heads: int = 12        # attention heads (d_model % num_heads == 0)
    seq_len: int = 64          # training context length (shorter = less memory)
    max_len: int = 1024         # generation max context
    n_layers: int = 12         # number of transformer blocks
    lr: float = 3e-4           # base learning rate (may be overridden per profile)

    # Optimization & scheduling
    max_epochs: int = 3
    patience: int = 4
    improvement_threshold: float = 0.1
    batch_size: int = 24
    epsilon: float = 1e-5
    gradAccumSteps: int = 24          # effective batch = batch_size * gradAccumSteps
    eval_every_steps: int = 100
    max_batches: int = 250
    save_every_steps: int = 500
    label_smoothing: float = 0.0
    dropout: float = 0.0

    # Adafactor-specific
    grad_clip: float = 0.9
    
    # Token accounting
    # (dynamic: recomputed in train.py; used for diagnostics/logging only)
    target_warmup_tokens: float = 2e7  # ≈ 20 M tokens of warmup (1-3% of total token count) 
    tokens_per_opt_step: int = 0

    # Debug
    debug: bool = True
    debug_every: int = 20
    log_random_sample: bool = False
    random_samples: int = 250

# Profile overrides ---------------------------------------------------------
@dataclass
class DebugProfile(BaseConfig):
    # fast iteration / overfit checks
    seq_len: int = 64
    n_layers: int = 2
    d_model: int = 512
    hidden_size: int = 1536
    batch_size: int = 4
    gradAccumSteps: int = 1
    lr: float = 5e-4
    eval_every_steps: int = 200
    save_every_steps: int = 1000
    max_batches: int = 50
    
    dropout: float = 0.0
    debug: bool = True
    debug_every: int = 50

@dataclass
class LocalProfile(BaseConfig):
    # target for M4 Pro, ~200M param model training locally
    seq_len: int = 1024
    n_layers: int = 12
    d_model: int = 768
    hidden_size: int = 2048
    batch_size: int = 14
    gradAccumSteps: int = 14         # effective batch ~196
    lr: float = 5e-4
    
    eval_every_steps: int = 50
    save_every_steps: int = 250
    max_batches: int = 250
    dropout: float = 0.1
    label_smoothing: float = 0.05
    debug: bool = True
    debug_every: int = 10

@dataclass
class LoRAProfile(BaseConfig):
    # LoRA finetuning: only adapter params updated so memory / batch can be bigger
    seq_len: int = 256
    n_layers: int = 12
    d_model: int = 768
    hidden_size: int = 2048
    batch_size: int = 4
    gradAccumSteps: int = 8          # effective batch = 32
    lr: float = 2e-3                 # LoRA often uses higher LR
    
    eval_every_steps: int = 250
    save_every_steps: int = 1_000
    dropout: float = 0.0
    debug: bool = False

# Export the chosen Config
if ACTIVE == "debug":
    Config = DebugProfile()
elif ACTIVE == "lora":
    Config = LoRAProfile()
else:
    Config = LocalProfile()

# Backwards compatibility: expose lowercase name if some modules import 'config'
config = Config
