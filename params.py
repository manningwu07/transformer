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
    lr: float = 5e-5 # Was 3e-4 but too slow, so bumping it up to x2 training speed. 

    # Optimization
    max_epochs: int = 5 # number of passes through the dataset
    patience: int = 16
    improvement_threshold: float = 0.01
    batch_size: int = 8
    epsilon: int = 1e-5
    gradAccumSteps: int = 16 # batchsize * gradAccumSteps = effective batchsize
    eval_every_steps = 500
    save_every_steps = 2000

    # Adam/Optimizer
    warmup_steps: int = 100
    decay_steps: int = 10_000
    adam_beta1: float = 0.90
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    max_batches: int = 250
    dropout: float = 0.05

    # Debug
    debug: bool = True
    debug_every: int = 200     # print every N steps
    log_random_sample: bool = False
    random_samples: int = 250


Config = TrainingConfig()
