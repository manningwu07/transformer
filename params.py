# params.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Core transformer parameters (MUST MATCH OR ELSE IT WILL FAIL)
    d_model: int = 768         # model width 
    hidden_size: int = 2048    # MLP hidden dim 
    vocab_size: int = 32_768   # |V| 
    num_heads: int = 12 # attention heads 
    seq_len: int = 512        # context length 
    max_len: int = 512
    n_layers: int = 8          # number of transformer blocks 
    lr: float = 1e-3 # 3e-3 for finetuning (LoRA)

    # Optimization
    max_epochs: int = 5 # number of passes through the dataset
    patience: int = 10
    improvement_threshold: float = 0.01
    batch_size: int = 256
    epsilon: int = 1e-4
    gradAccumSteps: int = 2 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 1_000
    max_batches: int = 500
    save_every_steps = 10_000
    label_smoothing: float = 0.0
    dropout: float = 0.05

    # AdaFactor
    warmup_steps: int = 5_000
    decay_steps: int = 50_000
    grad_clip: float = 1.0

    # Debug
    debug: bool = True
    debug_every: int = 500     # print every N steps
    log_random_sample: bool = False
    random_samples: int = 250


Config = TrainingConfig()
