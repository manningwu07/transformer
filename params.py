# params.py
from dataclasses import dataclass

# Choose a profile by setting ACTIVE = "local" / "debug" / "lora"
ACTIVE = "local"

@dataclass
class BaseConfig:
    # Core transformer parameters (must match any loaded checkpoint)
    d_model: int = 512         # model width
    hidden_size: int = 1536    # MLP hidden dim (typically 2-4x d_model)
    vocab_size: int = 65_536   # |V|
    num_heads: int = 8        # attention heads (d_model % num_heads == 0)
    seq_len: int = 64          # training context length (shorter = less memory)
    max_len: int = 1024         # generation max context
    n_layers: int = 8         # number of transformer blocks

    # Optimization & scheduling
    max_epochs: int = 3
    patience: int = 3
    improvement_threshold: float = 0.05
    batch_size: int = 24
    epsilon: float = 1e-4
    gradAccumSteps: int = 24          # effective batch = batch_size * gradAccumSteps
    eval_every_steps: int = 100
    max_batches: int = 250
    label_smoothing: float = 0.01
    dropout: float = 0.01

    # Adafactor-specific
    grad_clip: float = 1.0
    startLr: float = 3e-4
    endLr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.01
    totalOptSteps: int = 1_500
    
    # Token accounting
    # (dynamic: recomputed in train.py; used for diagnostics/logging only)
    target_warmup_tokens: float = 2e7  # ≈ 20M tokens of warmup (1-3% of total token count)

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
    eval_every_steps: int = 250
    max_batches: int = 100
    
    dropout: float = 0.0
    debug: bool = True
    debug_every: int = 100

@dataclass
class LocalProfile(BaseConfig):
    seq_len: int = 1024
    n_layers: int = 8
    d_model: int = 512
    hidden_size: int = 1536
    batch_size: int = 5             # Dont try to max out this bc it will be slow... leave some memory for other processes
    gradAccumSteps: int = 30         # effective batch ~150
    
    eval_every_steps: int = 250
    max_batches: int = 100
    dropout: float = 0.01
    label_smoothing: float = 0.01
    debug: bool = True
    debug_every: int = 33
    log_random_sample: bool = False
    random_samples: int = 250

@dataclass
class LoRAProfile(BaseConfig):
    # LoRA finetuning: only adapter params updated so memory / batch can be bigger
    seq_len: int = 256
    n_layers: int = 12
    d_model: int = 768
    hidden_size: int = 2048
    batch_size: int = 10
    gradAccumSteps: int = 10          # effective batch = 64
    lr: float = 1e-3
    
    eval_every_steps: int = 250
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
