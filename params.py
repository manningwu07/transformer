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
    lr: float = 6e-4 # 3e-3 for finetuning (LoRA)

    # Optimization
    max_epochs: int = 5 # number of passes through the dataset
    patience: int = 10
    improvement_threshold: float = 0.02
    batch_size: int = 12
    epsilon: int = 5e-5
    gradAccumSteps: int = 12 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 2_000
    max_batches: int = 250
    save_every_steps = 10_000
    label_smoothing: float = 0.005
    dropout: float = 0.05

    # AdaFactor
    warmup_steps: int = 5_000
    decay_steps: int = 150_000
    grad_clip: float = 1.0

    # Debug
    debug: bool = True
    debug_every: int = 500     # print every N steps
    log_random_sample: bool = True
    random_samples: int = 250


Config = TrainingConfig()
